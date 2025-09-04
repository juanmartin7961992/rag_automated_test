from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)



WRITE_CONFIG = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
READ_CONFIG = {"configurable": {"thread_id": "1"}}


def setup_db_checkpoint_table(db_uri: str):
    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        # call .setup() the first time you're using the checkpointer
        checkpointer.setup()
        checkpoint = {
            "v": 2,
            "ts": "2024-07-31T20:14:19.804150+00:00",
            "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            "channel_values": {
                "my_key": "meow",
                "node": "node"
            },
            "channel_versions": {
                "__start__": 2,
                "my_key": 3,
                "start:node": 3,
                "node": 3
            },
            "versions_seen": {
                "__input__": {},
                "__start__": {
                "__start__": 1
                },
                "node": {
                "start:node": 2
                }
            },
            "pending_sends": [],
        }

        # store checkpoint
        checkpointer.put(WRITE_CONFIG, checkpoint, {}, {})

        # load checkpoint
        checkpointer.get(READ_CONFIG)
        logger.info("**PostgresSaver setup completed. Checkpoints table exists.**")


async def asetup_db_checkpoint_table(db_uri: str):
    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
        checkpoint = {
            "v": 2,
            "ts": "2024-07-31T20:14:19.804150+00:00",
            "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            "channel_values": {
                "my_key": "meow",
                "node": "node"
            },
            "channel_versions": {
                "__start__": 2,
                "my_key": 3,
                "start:node": 3,
                "node": 3
            },
            "versions_seen": {
                "__input__": {},
                "__start__": {
                "__start__": 1
                },
                "node": {
                "start:node": 2
                }
            },
            "pending_sends": [],
        }

        # store checkpoint
        await checkpointer.aput(WRITE_CONFIG, checkpoint, {}, {})

        # load checkpoint
        await checkpointer.aget(READ_CONFIG)
        logger.info("**PostgresSaver setup completed. Checkpoints table exists.**")


def get_thread_history_from_db(db_uri: str, thread_id: str):
    """
    Retrieve thread history from database using PostgresSaver.
    
    Args:
        db_uri: PostgreSQL connection string
        thread_id: Thread ID to look up
        
    Returns:
        Messages from the thread if found, empty list otherwise
    """
    import logging
    logger = logging.getLogger("db_utils")
    logger.setLevel(logging.DEBUG)
    
    logger.debug(f"Retrieving thread history for {thread_id} using PostgresSaver")
    
    try:
        # Use PostgresSaver to retrieve the thread
        with PostgresSaver.from_conn_string(db_uri) as saver:
            # Format the thread ID in the way LangGraph expects it
            thread_config = {"configurable": {"thread_id": thread_id}}
            state = saver.get(thread_config)
            
            if state and "channel_values" in state:
                if "messages" in state["channel_values"]:
                    messages = state["channel_values"]["messages"]
                    logger.debug(f"Found {len(messages)} messages in thread")
                    return messages
        
        # If not in standard format, try direct thread_id lookup
        logger.debug(f"Standard lookup failed, trying direct thread_id={thread_id}")
        with PostgresSaver.from_conn_string(db_uri) as saver:
            # This assumes the thread_id in the database matches the one provided
            direct_config = {"thread_id": thread_id}
            state = saver.get(direct_config)
            
            if state and "channel_values" in state:
                if "messages" in state["channel_values"]:
                    messages = state["channel_values"]["messages"]
                    logger.debug(f"Found {len(messages)} messages with direct lookup")
                    return messages
        
        logger.debug(f"No messages found for thread {thread_id}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving thread from database: {str(e)}")
        return []
