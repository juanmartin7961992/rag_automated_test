from typing import Any, List, Optional, Sequence, Tuple, Union, Callable, Iterable, Dict, Literal, ClassVar, Type
from typing import (
    cast as typing_cast,
)
from typing_extensions import override
from collections.abc import Collection
import numpy as np
import asyncio
import sqlalchemy
from pydantic import ConfigDict, Field, model_validator
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever, LangSmithRetrieverParams
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_postgres.vectorstores import PGVector
from langchain_postgres._utils import maximal_marginal_relevance
from pgvector.sqlalchemy import HALFVEC, SPARSEVEC
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID, insert
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import (
    Session,
    declarative_base,
    relationship,
)
from sklearn.feature_extraction.text import CountVectorizer
import uuid
import logging
import enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


class DistanceStrategySymbols(str, enum.Enum):
    """Enumerator of the Distance strategy symbols."""

    EUCLIDEAN = "<->"
    COSINE = "<=>"
    MAX_INNER_PRODUCT = "<#>"


_LANGCHAIN_DEFAULT_COLLECTION_NAME = "dual_lanchain_db"
DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

Base = declarative_base()  # type: Any

_classes: Any = None


def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


def _create_vector_extension(conn: Connection) -> None:
    statement = sqlalchemy.text(
        "SELECT pg_advisory_xact_lock(1573678846307946496);"
        "CREATE EXTENSION IF NOT EXISTS vector;"
    )
    conn.execute(statement)
    conn.commit()


DBConnection = Union[sqlalchemy.engine.Engine, str]


def _get_dual_embedding_collection_store(
        vector_dimension: Optional[int] = None,
        sparse_vector_dimension: Optional[int] = None,
        ) -> Any:
    global _classes
    if _classes is not None:
        return _classes

    class CollectionStore(Base):
        """Collection store."""

        __tablename__ = "langchain_dual_pg_collection"

        uuid = sqlalchemy.Column(
            UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
        )
        name = sqlalchemy.Column(sqlalchemy.String, nullable=False, unique=True)
        cmetadata = sqlalchemy.Column(JSON)

        embeddings = relationship(
            "EmbeddingStore",
            back_populates="collection",
            passive_deletes=True,
        )

        @classmethod
        def get_by_name(
            cls, session: Session, name: str
        ) -> Optional["CollectionStore"]:
            return (
                session.query(cls)
                .filter(typing_cast(sqlalchemy.Column, cls.name) == name)
                .first()
            )

        @classmethod
        async def aget_by_name(
            cls, session: AsyncSession, name: str
        ) -> Optional["CollectionStore"]:
            return (
                (
                    await session.execute(
                        select(CollectionStore).where(
                            typing_cast(sqlalchemy.Column, cls.name) == name
                        )
                    )
                )
                .scalars()
                .first()
            )

        @classmethod
        def get_or_create(
            cls,
            session: Session,
            name: str,
            cmetadata: Optional[dict] = None,
        ) -> Tuple["CollectionStore", bool]:
            """Get or create a collection.
            Returns:
                 Where the bool is True if the collection was created.
            """  # noqa: E501
            created = False
            collection = cls.get_by_name(session, name)
            if collection:
                return collection, created

            collection = cls(name=name, cmetadata=cmetadata)
            session.add(collection)
            session.commit()
            created = True
            return collection, created

        @classmethod
        async def aget_or_create(
            cls,
            session: AsyncSession,
            name: str,
            cmetadata: Optional[dict] = None,
        ) -> Tuple["CollectionStore", bool]:
            """
            Get or create a collection.
            Returns [Collection, bool] where the bool is True if the collection was created.
            """  # noqa: E501
            created = False
            collection = await cls.aget_by_name(session, name)
            if collection:
                return collection, created

            collection = cls(name=name, cmetadata=cmetadata)
            session.add(collection)
            await session.commit()
            created = True
            return collection, created

    class EmbeddingStore(Base):
        """Embedding store."""

        __tablename__ = "langchain_dual_pg_embedding"

        id = sqlalchemy.Column(
            sqlalchemy.String, nullable=True, primary_key=True, index=True, unique=True
        )

        collection_id = sqlalchemy.Column(
            UUID(as_uuid=True),
            sqlalchemy.ForeignKey(
                f"{CollectionStore.__tablename__}.uuid",
                ondelete="CASCADE",
            ),
        )
        collection = relationship(CollectionStore, back_populates="embeddings")

        embedding: HALFVEC = sqlalchemy.Column(HALFVEC(vector_dimension))
        sparse_embedding: SPARSEVEC = sqlalchemy.Column(SPARSEVEC(sparse_vector_dimension))
        document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
        cmetadata = sqlalchemy.Column(JSONB, nullable=True)


        __table_args__ = (
            sqlalchemy.Index(
                "ix_dual_cmetadata_gin",
                "cmetadata",
                postgresql_using="gin",
                postgresql_ops={"cmetadata": "jsonb_path_ops"},
            ),
            # Create indexes for vector similarity search
            sqlalchemy.Index(
                "ix_dense_embedding_hnsw", 
                "embedding",
                postgresql_using="hnsw",
                postgresql_ops={"embedding": "halfvec_cosine_ops"},
            ),
            sqlalchemy.Index(
                "ix_sparse_embedding_hnsw", 
                "sparse_embedding",
                postgresql_using="hnsw",
                postgresql_ops={"sparse_embedding": "sparsevec_cosine_ops"},
            ),
        )

    _classes = (EmbeddingStore, CollectionStore)

    return _classes


class DualPGVector(PGVector):
    """
    PGVector extension that supports both dense and sparse embeddings.
    It extends the standard PGVector class to support:
    1. Dense embeddings (original functionality)
    2. Sparse embeddings (using pgvector's SPARSEVEC)
    """
    
    def __init__(
        self,
        embeddings: Embeddings,
        sparse_encoder: CountVectorizer,
        *,
        connection: Union[None, str, sqlalchemy.engine.Engine] = None,
        embedding_length: Optional[int] = None,
        sparse_embedding_length: Optional[int] = None,
        collection_name: str = "dual_lanchain_db",
        collection_metadata: Optional[dict] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        sparse_distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        engine_args: Optional[dict[str, Any]] = None,
        use_jsonb: bool = True,
        create_extension: bool = True,
        async_mode: bool = False,
    ) -> None:
        """
        Initialize the HybridPGVector store with both dense and sparse embedding support.

        Args:
            sparse_encoder: An instance of a fitted CountVectorizer from sklearn
                that will be used to convert text into sparse vectors.
            sparse_embedding_length: The length of the sparse embedding vector.
            sparse_distance_strategy: The distance strategy to use for sparse vectors.
                Options: 'cosine', 'inner', 'l2'. Default is 'cosine'.
            
            All other parameters are the same as the parent PGVector class.
        """
        self.sparse_encoder = sparse_encoder
        self._sparse_embedding_length = sparse_embedding_length
        self._sparse_distance_strategy = sparse_distance_strategy
        
        # Call parent constructor
        super().__init__(
            embeddings=embeddings,
            connection=connection,
            embedding_length=embedding_length, 
            collection_name=collection_name,
            collection_metadata=collection_metadata,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            logger=logger,
            relevance_score_fn=relevance_score_fn,
            engine_args=engine_args,
            use_jsonb=use_jsonb,
            create_extension=create_extension,
            async_mode=async_mode
        )
    
    def __post_init__(self) -> None:
        """Initialize the store with custom schema for sparse vectors."""
        if self.async_mode:
            raise ValueError("This method cannot be called in async mode. Use __apost_init__ instead.")
            
        if self.create_extension:
            self.create_vector_extension()

        # Use our custom embedding store that includes sparse vectors
        EmbeddingStore, CollectionStore = _get_dual_embedding_collection_store(
            self._embedding_length,
            self._sparse_embedding_length,
        )
        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
        
        # Create tables and collection
        self.create_tables_if_not_exists()
        self.create_collection()
    
    async def __apost_init__(self) -> None:
        """Async initialize the store with custom schema for sparse vectors."""
        if self._async_init:  # Warning: possible race condition
            return
        self._async_init = True

        # Use our custom embedding store that includes sparse vectors
        EmbeddingStore, CollectionStore = _get_dual_embedding_collection_store(
            self._embedding_length,
            self._sparse_embedding_length,
        )
        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
        
        if self.create_extension:
            await self.acreate_vector_extension()

        await self.acreate_tables_if_not_exists()
        await self.acreate_collection()

    @property
    def sparse_embeddings(self) -> CountVectorizer:
        return self.sparse_encoder
    
    @property
    def sparse_distance_strategy(self) -> Any:
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self.EmbeddingStore.sparse_embedding.l2_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            return self.EmbeddingStore.sparse_embedding.cosine_distance
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self.EmbeddingStore.sparse_embedding.max_inner_product
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )
        
    def create_vector_extension(self) -> None:
        assert self._engine, "engine not found"
        try:
            with self._engine.connect() as conn:
                _create_vector_extension(conn)
        except Exception as e:
            raise Exception(f"Failed to create vector extension: {e}") from e

    async def acreate_vector_extension(self) -> None:
        assert self._async_engine, "_async_engine not found"

        async with self._async_engine.begin() as conn:
            await conn.run_sync(_create_vector_extension)

    def create_tables_if_not_exists(self) -> None:
        with self._make_sync_session() as session:
            Base.metadata.create_all(session.get_bind())
            session.commit()

    async def acreate_tables_if_not_exists(self) -> None:
        assert self._async_engine, "This method must be called with async_mode"
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def drop_tables(self) -> None:
        with self._make_sync_session() as session:
            Base.metadata.drop_all(session.get_bind())
            session.commit()

    async def adrop_tables(self) -> None:
        assert self._async_engine, "This method must be called with async_mode"
        await self.__apost_init__()  # Lazy async init
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        with self._make_sync_session() as session:
            self.CollectionStore.get_or_create(
                session, self.collection_name, cmetadata=self.collection_metadata
            )
            session.commit()

    async def acreate_collection(self) -> None:
        await self.__apost_init__()  # Lazy async init
        async with self._make_async_session() as session:
            if self.pre_delete_collection:
                await self._adelete_collection(session)
            await self.CollectionStore.aget_or_create(
                session, self.collection_name, cmetadata=self.collection_metadata
            )
            await session.commit()

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        sparse_embeddings: List[List[float]],
        embedding: Embeddings,
        sparse_encoder: CountVectorizer,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        connection: Optional[str] = None,
        pre_delete_collection: bool = False,
        *,
        use_jsonb: bool = True,
        **kwargs: Any,
    ) -> 'DualPGVector':
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        store = cls(
            connection=connection,
            collection_name=collection_name,
            embeddings=embedding,
            sparse_encoder=sparse_encoder,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            use_jsonb=use_jsonb,
            **kwargs,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, sparse_embeddings=sparse_embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    @classmethod
    async def __afrom(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        sparse_embeddings: List[List[float]],
        embedding: Embeddings,
        sparse_encoder: CountVectorizer,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        connection: Optional[str] = None,
        pre_delete_collection: bool = False,
        *,
        use_jsonb: bool = True,
        **kwargs: Any,
    ) -> 'DualPGVector':
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        store = cls(
            connection=connection,
            collection_name=collection_name,
            embeddings=embedding,
            sparse_encoder=sparse_encoder,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            use_jsonb=use_jsonb,
            async_mode=True,
            **kwargs,
        )

        await store.aadd_embeddings(
            texts=texts, embeddings=embeddings, sparse_embeddings=sparse_embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    @classmethod
    def from_texts(
        cls: Type[PGVector],
        texts: List[str],
        embedding: Embeddings,
        sparse_encoder: CountVectorizer,
        metadatas: Optional[List[dict]] = None,
        *,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        use_jsonb: bool = True,
        **kwargs: Any,
    ) -> 'DualPGVector':
        """Return VectorStore initialized from documents and embeddings."""
        embeddings = embedding.embed_documents(list(texts))
        sparse_embeddings = sparse_encoder._get_sparse_vector(list(texts))
        return cls.__from(
            texts,
            embeddings,
            sparse_embeddings,
            embedding,
            sparse_encoder,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            use_jsonb=use_jsonb,
            **kwargs,
        )

    @classmethod
    async def afrom_texts(
        cls: Type[PGVector],
        texts: List[str],
        embedding: Embeddings,
        sparse_encoder: CountVectorizer,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        *,
        use_jsonb: bool = True,
        **kwargs: Any,
    ) -> 'DualPGVector':
        """Return VectorStore initialized from documents and embeddings."""
        embeddings = await embedding.aembed_documents(list(texts))
        sparse_embeddings = await sparse_encoder._aget_sparse_vector(list(texts))
        return await cls.__afrom(
            texts,
            embeddings,
            sparse_embeddings,
            embedding,
            sparse_encoder,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            use_jsonb=use_jsonb,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float], List[float]]],
        embedding: Embeddings,
        sparse_encoder: CountVectorizer,
        *,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> 'DualPGVector':
        """Construct PGVector wrapper from raw documents and embeddings."""
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]
        sparse_embeddings = [t[2] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            sparse_embeddings,
            embedding,
            sparse_encoder,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    async def afrom_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float], List[float]]],
        embedding: Embeddings,
        sparse_encoder: CountVectorizer,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> 'DualPGVector':
        """Construct PGVector wrapper from raw documents and pre-
        generated embeddings."""
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]
        sparse_embeddings = [t[2] for t in text_embeddings]

        return await cls.__afrom(
            texts,
            embeddings,
            sparse_embeddings,
            embedding,
            sparse_encoder,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )
    
    @classmethod
    def from_documents(
        cls: Type[PGVector],
        documents: List[Document],
        embedding: Embeddings,
        sparse_encoder: CountVectorizer,
        *,
        connection: Optional[DBConnection] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        use_jsonb: bool = True,
        **kwargs: Any,
    ) -> 'DualPGVector':
        """Return VectorStore initialized from documents and embeddings."""

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        return cls.from_texts(
            texts=texts,
            pre_delete_collection=pre_delete_collection,
            embedding=embedding,
            sparse_encoder=sparse_encoder,
            distance_strategy=distance_strategy,
            metadatas=metadatas,
            connection=connection,
            ids=ids,
            collection_name=collection_name,
            use_jsonb=use_jsonb,
            **kwargs,
        )

    @classmethod
    async def afrom_documents(
        cls: Type[PGVector],
        documents: List[Document],
        embedding: Embeddings,
        sparse_encoder: CountVectorizer,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        *,
        use_jsonb: bool = True,
        **kwargs: Any,
    ) -> 'DualPGVector':
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.get_connection_string(kwargs)

        kwargs["connection"] = connection_string

        return await cls.afrom_texts(
            texts=texts,
            pre_delete_collection=pre_delete_collection,
            embedding=embedding,
            sparse_encoder=sparse_encoder,
            distance_strategy=distance_strategy,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            use_jsonb=use_jsonb,
            **kwargs,
        )
    
    def _get_sparse_vector(self, text: List[str] | str) -> List[List[float]] | List[float]:
        """
        Convert text to a sparse vector representation using the sparse encoder.
        
        Args:
            text: The text to convert to a sparse vector.
            
        Returns:
            A sparse vector representation of the text as a list.
        """
        if self.sparse_encoder is None:
            raise ValueError("Cannot perform sparse encoding without a sparse_encoder")
            
        # Get sparse vector from CountVectorizer
        if isinstance(text, str):
            return self.sparse_encoder.transform([text]).toarray()[0]
        else:
            return self.sparse_encoder.transform(text).toarray().tolist()
    
    async def _aget_sparse_vector(self, text: List[str] | str) -> List[List[float]] | List[float]:
        """
        Asynchronous version to convert text to a sparse vector representation.
        
        Args:
            text: The text to convert to a sparse vector.
            
        Returns:
            A sparse vector representation of the text as a list.
        """
        # Since the vectorization operation might be CPU-bound, 
        # we can use run_in_executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_sparse_vector, text)
    
    def add_embeddings(
        self,
        texts: Sequence[str],
        embeddings: List[List[float]],
        sparse_embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            sparse_embeddings: List of sparse embedding vectors.
            metadatas: List of metadatas associated with the texts.
            ids: Optional list of ids for the documents.
                 If not provided, will generate a new id for each document.
            kwargs: vectorstore specific parameters
        """
        assert not self._async_engine, "This method must be called with sync_mode"
        if ids is None:
            ids_ = [str(uuid.uuid4()) for _ in texts]
        else:
            ids_ = [id if id is not None else str(uuid.uuid4()) for id in ids]

        if not metadatas:
            metadatas = [{} for _ in texts]

        with self._make_sync_session() as session:  # type: ignore[arg-type]
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            data = [
                {
                    "id": id,
                    "collection_id": collection.uuid,
                    "embedding": embedding,
                    "sparse_embedding": sparse_embedding,
                    "document": text,
                    "cmetadata": metadata or {},
                }
                for text, metadata, embedding, sparse_embedding, id in zip(
                    texts, metadatas, embeddings, sparse_embeddings, ids_
                )
            ]
            stmt = insert(self.EmbeddingStore).values(data)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                # Conflict detection based on these columns
                set_={
                    "embedding": stmt.excluded.embedding,
                    "sparse_embedding": stmt.excluded.sparse_embedding,
                    "document": stmt.excluded.document,
                    "cmetadata": stmt.excluded.cmetadata,
                },
            )
            session.execute(on_conflict_stmt)
            session.commit()

        return ids_

    async def aadd_embeddings(
        self,
        texts: Sequence[str],
        embeddings: List[List[float]],
        sparse_embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            sparse_embeddings: List of sparse embedding vectors.
            metadatas: List of metadatas associated with the texts.
            ids: Optional list of ids for the texts.
                 If not provided, will generate a new id for each text.
            kwargs: vectorstore specific parameters
        """
        await self.__apost_init__()  # Lazy async init

        if ids is None:
            ids_ = [str(uuid.uuid4()) for _ in texts]
        else:
            ids_ = [id if id is not None else str(uuid.uuid4()) for id in ids]

        if not metadatas:
            metadatas = [{} for _ in texts]

        async with self._make_async_session() as session:  # type: ignore[arg-type]
            collection = await self.aget_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            data = [
                {
                    "id": id,
                    "collection_id": collection.uuid,
                    "embedding": embedding,
                    "sparse_embedding": sparse_embedding,
                    "document": text,
                    "cmetadata": metadata or {},
                }
                for text, metadata, embedding, sparse_embedding, id in zip(
                    texts, metadatas, embeddings, sparse_embeddings, ids_
                )
            ]
            stmt = insert(self.EmbeddingStore).values(data)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                # Conflict detection based on these columns
                set_={
                    "embedding": stmt.excluded.embedding,
                    "sparse_embedding": stmt.excluded.sparse_embedding,
                    "document": stmt.excluded.document,
                    "cmetadata": stmt.excluded.cmetadata,
                },
            )
            await session.execute(on_conflict_stmt)
            await session.commit()

        return ids_

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids for the texts.
                 If not provided, will generate a new id for each text.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        assert not self._async_engine, "This method must be called without async_mode"
        texts_ = list(texts)
        embeddings = self.embedding_function.embed_documents(texts_)
        sparse_embeddings = self.sparse_encoder.transform(texts_).toarray()
        return self.add_embeddings(
            texts=texts_,
            embeddings=list(embeddings),
            sparse_embeddings=list(sparse_embeddings),
            metadatas=list(metadatas) if metadatas else None,
            ids=list(ids) if ids else None,
            **kwargs,
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids for the texts.
                 If not provided, will generate a new id for each text.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        await self.__apost_init__()  # Lazy async init
        texts_ = list(texts)
        embeddings = await self.embedding_function.aembed_documents(texts_)
        sparse_embeddings = self.sparse_encoder.transform(texts_).toarray()
        return await self.aadd_embeddings(
            texts=texts_,
            embeddings=list(embeddings),
            sparse_embeddings=list(sparse_embeddings),
            metadatas=list(metadatas) if metadatas else None,
            ids=list(ids) if ids else None,
            **kwargs,
        )
    
    @classmethod
    def from_existing_index(
        cls: Type['DualPGVector'],
        embeddings: Embeddings,
        sparse_encoder: CountVectorizer,
        *,
        collection_name: str = "dual_lanchain_db",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        connection: Optional[DBConnection] = None,
        **kwargs: Any,
    ) -> PGVector:
        """
        Get instance of an existing PGVector store.This method will
        return the instance of the store without inserting any new
        embeddings
        """
        store = cls(
            connection=connection,
            collection_name=collection_name,
            embeddings=embeddings,
            sparse_encoder=sparse_encoder,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

        return store

    @classmethod
    async def afrom_existing_index(
        cls: Type['DualPGVector'],
        embeddings: Embeddings,
        sparse_encoder: CountVectorizer,
        *,
        collection_name: str = "dual_lanchain_db",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        connection: Optional[DBConnection] = None,
        **kwargs: Any,
    ) -> PGVector:
        """
        """
        store = DualPGVector(
            connection=connection,
            collection_name=collection_name,
            embeddings=embeddings,
            sparse_encoder=sparse_encoder,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            async_mode=True,
            **kwargs,
        )

        return store
    
    def __query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        embedding_type: Literal["dense", "sparse"] = "dense",
    ) -> Sequence[Any]:
        """Query the collection."""
        with self._make_sync_session() as session:  # type: ignore[arg-type]
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            filter_by = [self.EmbeddingStore.collection_id == collection.uuid]
            if filter:
                if self.use_jsonb:
                    filter_clauses = self._create_filter_clause(filter)
                    if filter_clauses is not None:
                        filter_by.append(filter_clauses)
                else:
                    # Old way of doing things
                    filter_clauses = self._create_filter_clause_json_deprecated(filter)
                    filter_by.extend(filter_clauses)

            _type = self.EmbeddingStore

            if embedding_type == "dense":
                query_distance_strategy = self.distance_strategy
            else:
                query_distance_strategy = self.sparse_distance_strategy

            results: List[Any] = (
                session.query(
                    self.EmbeddingStore,
                    query_distance_strategy(embedding).label("distance"),
                )
                .filter(*filter_by)
                .order_by(sqlalchemy.asc("distance"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
                .all()
            )

        return results

    async def __aquery_collection(
        self,
        session: AsyncSession,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        embedding_type: Literal["dense", "sparse"] = "dense",
    ) -> Sequence[Any]:
        """Query the collection."""
        async with self._make_async_session() as session:  # type: ignore[arg-type]
            collection = await self.aget_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            filter_by = [self.EmbeddingStore.collection_id == collection.uuid]
            if filter:
                if self.use_jsonb:
                    filter_clauses = self._create_filter_clause(filter)
                    if filter_clauses is not None:
                        filter_by.append(filter_clauses)
                else:
                    # Old way of doing things
                    filter_clauses = self._create_filter_clause_json_deprecated(filter)
                    filter_by.extend(filter_clauses)

            _type = self.EmbeddingStore

            if embedding_type == "dense":
                query_distance_strategy = self.distance_strategy
            else:
                query_distance_strategy = self.sparse_distance_strategy

            stmt = (
                select(
                    self.EmbeddingStore,
                    query_distance_strategy(embedding).label("distance"),
                )
                .filter(*filter_by)
                .order_by(sqlalchemy.asc("distance"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
            )

            results: Sequence[Any] = (await session.execute(stmt)).all()

            return results
        
    def sparse_similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with PGVector with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        assert not self._async_engine, "This method must be called without async_mode"
        embedding = self._get_sparse_vector(query)
        return self.sparse_similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    async def asparse_similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with PGVector with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        await self.__apost_init__()  # Lazy async init
        embedding = await self._aget_sparse_vector(query)
        return await self.asparse_similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def sparse_similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        assert not self._async_engine, "This method must be called without async_mode"
        embedding = self._get_sparse_vector(query)
        docs = self.sparse_similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    async def asparse_similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        await self.__apost_init__()  # Lazy async init
        embedding = self._get_sparse_vector(query)
        docs = await self.asparse_similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs
    
    def sparse_similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        assert not self._async_engine, "This method must be called without async_mode"
        docs_and_scores = self.sparse_similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return _results_to_docs(docs_and_scores)

    async def asparse_similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        assert self._async_engine, "This method must be called with async_mode"
        await self.__apost_init__()  # Lazy async init
        docs_and_scores = await self.asparse_similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return _results_to_docs(docs_and_scores)
    
    def sparse_similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        assert not self._async_engine, "This method must be called without async_mode"
        results = self.__query_collection(embedding=embedding, k=k, filter=filter, embedding_type="sparse")

        return self._results_to_docs_and_scores(results)

    async def asparse_similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        await self.__apost_init__()  # Lazy async init
        async with self._make_async_session() as session:  # type: ignore[arg-type]
            results = await self.__aquery_collection(
                session=session, embedding=embedding, k=k, filter=filter, embedding_type="sparse"
            )

            return self._results_to_docs_and_scores(results)
        
    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Original PGVector method, only extracting the Postgres embeddings"""
        assert not self._async_engine, "This method must be called without async_mode"
        from pgvector.utils.halfvec import HalfVector

        results = self.__query_collection(embedding=embedding, k=fetch_k, filter=filter)

        embedding_list = [result.EmbeddingStore.embedding for result in results]
        if isinstance(embedding_list[0], HalfVector):
            embedding_list = [embedding._value for embedding in embedding_list]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = self._results_to_docs_and_scores(results)

        return [r for i, r in enumerate(candidates) if i in mmr_selected]

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Original PGVector method, only extracting the Postgres embeddings"""
        await self.__apost_init__()  # Lazy async init
        from pgvector.utils.halfvec import HalfVector

        async with self._make_async_session() as session:
            results = await self.__aquery_collection(
                session=session, embedding=embedding, k=fetch_k, filter=filter
            )

            embedding_list = [result.EmbeddingStore.embedding for result in results]
            if isinstance(embedding_list[0], HalfVector):
                embedding_list = [embedding._value for embedding in embedding_list]

            mmr_selected = maximal_marginal_relevance(
                np.array(embedding, dtype=np.float32),
                embedding_list,
                k=k,
                lambda_mult=lambda_mult,
            )

            candidates = self._results_to_docs_and_scores(results)

            return [r for i, r in enumerate(candidates) if i in mmr_selected]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Original PGVector method, only extracting the Postgres embeddings"""
        embedding = self.embeddings.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Original PGVector method, only extracting the Postgres embeddings"""
        await self.__apost_init__()  # Lazy async init
        embedding = await self.embeddings.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Original PGVector method, only extracting the Postgres embeddings"""
        embedding = self.embeddings.embed_query(query)
        docs = self.max_marginal_relevance_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

    async def amax_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Original PGVector method, only extracting the Postgres embeddings"""
        await self.__apost_init__()  # Lazy async init
        embedding = await self.embeddings.aembed_query(query)
        docs = await self.amax_marginal_relevance_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Original PGVector method, only extracting the Postgres embeddings"""
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

        return _results_to_docs(docs_and_scores)

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Original PGVector method, only extracting the Postgres embeddings"""
        await self.__apost_init__()  # Lazy async init
        docs_and_scores = (
            await self.amax_marginal_relevance_search_with_score_by_vector(
                embedding,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
                **kwargs,
            )
        )

        return _results_to_docs(docs_and_scores)
    
    def max_marginal_and_sparse_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        sparse_k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        """
        embedding = self.embeddings.embed_query(query)
        sparse_embedding = self._get_sparse_vector(query)

        dense_results = self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        # Add IDs of dense results to filter
        dense_ids = [doc.metadata["id"] for doc in dense_results]
        if filter:
            filter["id"] = {"$nin": list(dense_ids)}
        else:
            filter = {"id": {"$nin": list(dense_ids)}}
        
        sparse_results = self.sparse_similarity_search_by_vector(
            embedding=sparse_embedding,
            k=sparse_k,
            filter=filter,
        )
        # Combine the results
        combined_results = dense_results + sparse_results

        return combined_results

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        sparse_k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Original PGVector method, only extracting the Postgres embeddings"""
        await self.__apost_init__()  # Lazy async init
        embedding = await self.embeddings.aembed_query(query)
        sparse_embedding = await self._aget_sparse_vector(query)

        dense_results = await self.amax_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        # Add IDs of dense results to filter
        dense_ids = [doc.metadata["id"] for doc in dense_results]
        if filter:
            filter["id"] = {"$nin": list(dense_ids)}
        else:
            filter = {"id": {"$nin": list(dense_ids)}}
        # Add IDs of dense results to filter
        dense_ids = [doc.metadata["id"] for doc in dense_results]

        sparse_results = await self.asparse_similarity_search_by_vector(
            embedding=sparse_embedding,
            k=sparse_k,
            filter=filter,
        )
        # Combine the results
        combined_results = dense_results + sparse_results
        return combined_results
    
    def dense_and_sparse_relevance_search(
        self,
        query: str,
        k: int = 4,
        sparse_k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        """
        embedding = self.embeddings.embed_query(query)
        sparse_embedding = self._get_sparse_vector(query)

        dense_results = self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        # Add IDs of dense results to filter
        dense_ids = [doc.metadata["id"] for doc in dense_results]
        if filter:
            filter["id"] = {"$nin": list(dense_ids)}
        else:
            filter = {"id": {"$nin": list(dense_ids)}}
        
        sparse_results = self.sparse_similarity_search_by_vector(
            embedding=sparse_embedding,
            k=sparse_k,
            filter=filter,
        )
        # Combine the results
        combined_results = dense_results + sparse_results

        return combined_results

    async def adense_and_sparse_relevance_search(
        self,
        query: str,
        k: int = 4,
        sparse_k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Original PGVector method, only extracting the Postgres embeddings"""
        await self.__apost_init__()  # Lazy async init
        embedding = await self.embeddings.aembed_query(query)
        sparse_embedding = await self._aget_sparse_vector(query)

        dense_results = await self.asimilarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        # Add IDs of dense results to filter
        dense_ids = [doc.metadata["id"] for doc in dense_results]
        if filter:
            filter["id"] = {"$nin": list(dense_ids)}
        else:
            filter = {"id": {"$nin": list(dense_ids)}}
        # Add IDs of dense results to filter
        dense_ids = [doc.metadata["id"] for doc in dense_results]

        sparse_results = await self.asparse_similarity_search_by_vector(
            embedding=sparse_embedding,
            k=sparse_k,
            filter=filter,
        )
        # Combine the results
        combined_results = dense_results + sparse_results
        return combined_results
    
    def _get_retriever_tags(self) -> list[str]:
        """Get tags for retriever."""
        tags = [self.__class__.__name__]
        if self.embeddings:
            tags.append(self.embeddings.__class__.__name__)
        return tags

    def as_retriever(self, **kwargs: Any) -> 'DualVectorStoreRetriever':
        """
        """
        tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()
        return DualVectorStoreRetriever(vectorstore=self, tags=tags, **kwargs)
    

class DualVectorStoreRetriever(BaseRetriever):
    """Base Retriever class for VectorStore."""

    vectorstore: DualPGVector
    """VectorStore to use for retrieval."""
    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
        "sparse_similarity",
        "dual_similarity",
        "dual_mmr"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_search_type(cls, values: dict) -> Any:
        """Validate search type.

        Args:
            values: Values to validate.

        Returns:
            Values: Validated values.

        Raises:
            ValueError: If search_type is not one of the allowed search types.
            ValueError: If score_threshold is not specified with a float value(0~1)
        """
        search_type = values.get("search_type", "similarity")
        if search_type not in cls.allowed_search_types:
            msg = (
                f"search_type of {search_type} not allowed. Valid values are: "
                f"{cls.allowed_search_types}"
            )
            raise ValueError(msg)
        if search_type == "similarity_score_threshold":
            score_threshold = values.get("search_kwargs", {}).get("score_threshold")
            if (score_threshold is None) or (not isinstance(score_threshold, float)):
                msg = (
                    "`score_threshold` is not specified with a float value(0~1) "
                    "in `search_kwargs`."
                )
                raise ValueError(msg)
        return values

    def _get_ls_params(self, **kwargs: Any) -> LangSmithRetrieverParams:
        """Get standard params for tracing."""
        _kwargs = self.search_kwargs | kwargs

        ls_params = super()._get_ls_params(**_kwargs)
        ls_params["ls_vector_store_provider"] = self.vectorstore.__class__.__name__

        if self.vectorstore.embeddings:
            ls_params["ls_embedding_provider"] = (
                self.vectorstore.embeddings.__class__.__name__
            )
        elif hasattr(self.vectorstore, "embedding") and isinstance(
            self.vectorstore.embedding, Embeddings
        ):
            ls_params["ls_embedding_provider"] = (
                self.vectorstore.embedding.__class__.__name__
            )

        return ls_params

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        _kwargs = self.search_kwargs | kwargs
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(query, **_kwargs)
        elif self.search_type == "sparse_similarity":
            docs = self.vectorstore.sparse_similarity_search(query, **_kwargs)
        elif self.search_type == "dual_similarity":
            docs = self.vectorstore.dense_and_sparse_relevance_search(
                query, **_kwargs
            )
        elif self.search_type == "dual_mmr":
            docs = self.vectorstore.max_marginal_and_sparse_relevance_search(
                query, **_kwargs
            )
        else:
            msg = f"search_type of {self.search_type} not allowed."
            raise ValueError(msg)
        return docs

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        _kwargs = self.search_kwargs | kwargs
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search(query, **_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, **_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **_kwargs
            )
        elif self.search_type == "sparse_similarity":
            docs = await self.vectorstore.asparse_similarity_search(query, **_kwargs)
        elif self.search_type == "dual_similarity":
            docs = await self.vectorstore.adense_and_sparse_relevance_search(
                query, **_kwargs
            )
        elif self.search_type == "dual_mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **_kwargs
            )
        else:
            msg = f"search_type of {self.search_type} not allowed."
            raise ValueError(msg)
        return docs

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add documents to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            List of IDs of the added texts.
        """
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(
        self, documents: list[Document], **kwargs: Any
    ) -> list[str]:
        """Async add documents to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            List of IDs of the added texts.
        """
        return await self.vectorstore.aadd_documents(documents, **kwargs)



if __name__ == "__main__":
    from langchain_openai import OpenAIEmbeddings
    from sklearn.feature_extraction.text import TfidfVectorizer
    import json
    from typing import Dict, Any, List
    import os
    from dotenv import load_dotenv

    load_dotenv()

    vectorizer = TfidfVectorizer()
    print("The OPENai_api_key is " + os.getenv("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))

    host = os.getenv("DB_HOST") 
    port = os.getenv("DB_PORT") 
    dbname = os.getenv("DB_NAME") 
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"



    def load_json(file_path: str) -> List[Dict[str, Any]]:
        """Load JSON file and return data."""
        with open(file_path, 'r') as f:
            return json.load(f)
        
    # embedding_data = load_json('/Users/iftacharbel/git/DLYMI/playground/data/embedding/merged_data.json')

    # import uuid

    # metadatas = [
    #         {
    #         'title': v['metadata'].get('title'), 
    #         'url': v['metadata'].get('file_url', v['metadata'].get('external_url', v['metadata'].get('youtube_url'))),
    #         'is_understood': True if 'www.understood.org' in v['metadata'].get('file_url', v['metadata'].get('external_url', v['metadata'].get('youtube_url'))) else False,
    #         'id': k.split('/')[-1]
    #     }
    #     for k, v in embedding_data.items()
    #     ]
    # # ids = [hashlib.md5(v['text'].encode()).hexdigest() for v in embedding_data.values()]
    # ids = [str(uuid.uuid3(uuid.NAMESPACE_DNS, v['text'])) for v in embedding_data.values()]

    # all_data = list(zip([v['text'] for v in embedding_data.values()], [v['embedding'] for v in embedding_data.values()], metadatas, ids))

    # # Sort by metadata['id']
    # all_data_sorted = sorted(all_data, key=lambda x: [int(i) for i in x[2]['id'].split('_')])
    # idx_to_remove = [idx for idx, i in enumerate(all_data_sorted) if i[2]['url'] == 'https://www.reel2e.org/_files/ugd/6c0e36_567b7a871d9b4fd99ba3f6c3ead91faa.pdf']

    # # Check if idx_to_remove is continuous
    # assert all(idx_to_remove[i] + 1 == idx_to_remove[i + 1] for i in range(len(idx_to_remove) - 1))
    # # Remove the badly parsed PDF
    # all_data_sorted = all_data_sorted[:idx_to_remove[0]] + all_data_sorted[idx_to_remove[-1] + 1:]

    # # Remove any duplications by IDs
    # all_data_unique = []
    # unique_ids_set = set()
    # unique_urls_map = dict()
    # for _text, embedding, metadata, id in all_data_sorted:
    #     if id not in unique_ids_set:
    #         unique_ids_set.add(id)
    #         # Check if the URL is unique
    #         url = metadata['url']
    #         if url not in unique_urls_map:
    #             unique_urls_map[url] = metadata['id'].split('_')[0]
    #             all_data_unique.append((_text, embedding, metadata, id))
    #         elif unique_urls_map[url] == metadata['id'].split('_')[0]:
    #             all_data_unique.append((_text, embedding, metadata, id))

    # assert len(all_data_unique) == 12242

    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer


    def load_vectorizer(filename: str) -> TfidfVectorizer:
        """
        Load the a TfidfVectorizer from a file.
        
        Parameters:
        - filename: The name of the file to load the vectorizer state from.
        """
        with open(filename, 'rb') as f:
            vectorizer, vectorizer_data = pickle.load(f)
            
        # Restore the internal attributes
        vectorizer.set_params(**vectorizer_data['params'])
        vectorizer.vocabulary_ = vectorizer_data['vocabulary_']
        vectorizer.idf_ = vectorizer_data['idf_']
        vectorizer._tfidf = vectorizer_data['_tfidf']
        
        return vectorizer
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ASSETS_PATH = os.path.join(BASE_DIR, "assets", "tfidf_vectorizer.pkl")

    sparse_encoder = load_vectorizer(ASSETS_PATH)

    vector_store = DualPGVector.from_existing_index(
        embeddings=embeddings,
        sparse_encoder=sparse_encoder,
        collection_name='dual_lanchain_db',
        connection=connection_string,
    )

    print("Loading json file....")
    # json_data = load_json('C:/Users/juanm/Downloads/digital_promise_to_upload_test_2.json')
    
    JSON_FILE = os.path.join(BASE_DIR, "assets", "digital_promise_to_upload.json")
    json_data = load_json(JSON_FILE)
    print("End loading json file....")

    print(f"Number of records: {len(json_data)}\n")

    BATCH_SIZE = 300
    total_items = len(json_data)

    for i in range(0, total_items, BATCH_SIZE):
        batch = json_data[i:i + BATCH_SIZE]

    
        # Extract texts and metadatas
        texts = []
        metadatas = []
        ids = []

        for item in batch:
            texts.append(item.get("text", ""))  
            metadatas.append(item.get("metadata", {}))
            ids.append(item.get("uuid"))

        # Print to verify
        # for text, metadata, id in zip(texts, metadatas, ids):
        #     print(f"Id: {id}")
        #     print(f"Metadata: {metadata}")
        #     print(f"Text: {text[:100]}...\n")  # Print first 100 chars

        # Add to vector store
        idsSaved = vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"Successfully Number ids added: {len(idsSaved)}")
        print(f"Successfully Ids added: {idsSaved} \n")

    # simliarity_search_embed = vector_store.max_marginal_relevance_search_by_vector(
    #     embedding=query_embeddings[0]['embedding'],
    #     k=5,
    #     feathch_k=10,
    #     lambda_mult=0.8,
    #     filter={'is_understood': False},
    # )

    # retriever = vector_store.as_retriever(search_type="dual_mmr", search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.8, "sparse_k": 5, 
    #                                                                              "filter": {'is_understood': False}})

    # sparse_retriever = vector_store.as_retriever(
    #     search_type="sparse_similarity",
    #     search_kwargs={"k": 100, "filter": {'is_understood': False}}
    #     )
    
    # res = sparse_retriever.invoke('Pinchi')
    # for i, doc in enumerate(res, 1):
    #     print(f"Result {i}:")
    #     print(f"Content: {doc.page_content}")
    #     print(f"Metadata: {doc.metadata}")
    #     print("-" * 40)
    

    # # Create the TF-IDF vectorizer and fit it on the text data
    # vectorizer = TfidfVectorizer()
    # texts = [v[0] for v in all_data_unique]
    # vectorizer.fit(texts)
    # # Get the sparse dim
    # sparse_embedding_length = len(vectorizer.get_feature_names_out())


#     vector_store = DualPGVector(
#         embeddings=embeddings,
#         sparse_encoder=vectorizer,
#         embedding_length=3072,
#         sparse_embedding_length=sparse_embedding_length,
#         collection_name='dual_lanchain_db',
#         connection=connection_string,
#         use_jsonb=True,
#         pre_delete_collection=True,
#     )

#     # Iterate over the data in batches and add to the vector store
#     batch_size = 2000
#     for i in range(0, len(all_data_unique), batch_size):
#         batch = all_data_unique[i:i + batch_size]
#         texts_batch = [v[0] for v in batch]
#         embeddings_batch = [v[1] for v in batch]
#         sparse_embeddings_batch = vectorizer.transform(texts_batch).toarray().tolist()
#         metadatas_batch = [v[2] for v in batch]
#         ids_batch = [v[3] for v in batch]

#         vector_store.add_embeddings(
#             texts=texts_batch,
#             embeddings=embeddings_batch,
#             sparse_embeddings=sparse_embeddings_batch,
#             metadatas=metadatas_batch,
#             ids=ids_batch,
#         )

#         print(f"Added batch {i // batch_size + 1} of {len(all_data_unique) // batch_size + 1} to the vector store.")
