import logging
import sys
import typing
from pymongo import MongoClient, errors
from src.configuration.config import settings
from src.exception.exceptions import DatabaseConnectionError, error_message_detail

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_mongo_client():
    try:
        mongo_uri = settings.mongodb_uri
        if not mongo_uri:
            logger.error('MONGODB_URI not set in environment variables!')
            raise DatabaseConnectionError('MONGODB_URI not set in environment variables!')
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        logger.info('Successfully connected to MongoDB.')
        return client
    except errors.ServerSelectionTimeoutError as e:
        error_message_detail(e, sys)
        logger.error(f'MongoDB connection timed out: {e}')
        raise DatabaseConnectionError('MongoDB connection timed out.')
    except Exception as e:
        error_message_detail(e, sys)
        logger.error(f'Unexpected error connecting to MongoDB: {e}')
        raise DatabaseConnectionError(str(e))

def get_collection(collection_name, db_name=None):
    client = get_mongo_client()
    db = client[db_name or settings.mongodb_db_name]
    return db[collection_name]

def fetch_stock_data(symbol, db_name=None):
    collection = get_collection('stocks', db_name)
    return list(collection.find({'symbol': symbol}))

def save_analysis(symbol, analysis_dict, db_name=None):
    collection = get_collection('analysis', db_name)
    collection.update_one({'symbol': symbol}, {'$set': analysis_dict}, upsert=True)

def fetch_holdings(symbol, db_name=None):
    collection = get_collection('holdings', db_name)
    return list(collection.find({'symbol': symbol}))

def save_holdings(symbol, holdings_dict, db_name=None):
    collection = get_collection('holdings', db_name)
    collection.update_one({'symbol': symbol}, {'$set': holdings_dict}, upsert=True)

# Add more CRUD functions as needed for other modules

def fetch_news(symbol: str, db_name: typing.Optional[str] = None) -> typing.List[typing.Dict[str, typing.Any]]:
    collection = get_collection('news', db_name)
    return list(collection.find({'symbol': symbol}, {'_id': 0}))

def save_news(symbol: str, news_list: typing.List[typing.Dict[str, typing.Any]], db_name: typing.Optional[str] = None) -> None:
    collection = get_collection('news', db_name)
    for article in news_list:
        doc = {**article, 'symbol': symbol}
        collection.update_one({'symbol': symbol, 'url': doc.get('url')}, {'$set': doc}, upsert=True)

def fetch_events(symbol: str, db_name: typing.Optional[str] = None) -> typing.List[typing.Dict[str, typing.Any]]:
    collection = get_collection('events', db_name)
    return list(collection.find({'symbol': symbol}, {'_id': 0}))

def save_events(symbol: str, events_list: typing.List[typing.Dict[str, typing.Any]], db_name: typing.Optional[str] = None) -> None:
    collection = get_collection('events', db_name)
    for event in events_list:
        doc = {**event, 'symbol': symbol}
        key = {'symbol': symbol, 'event_id': doc.get('event_id')} if doc.get('event_id') else {'symbol': symbol, 'timestamp': doc.get('timestamp')}
        collection.update_one(key, {'$set': doc}, upsert=True)

def fetch_portfolio(user_id: str, db_name: typing.Optional[str] = None) -> typing.Dict[str, typing.Any]:
    collection = get_collection('portfolios', db_name)
    return collection.find_one({'user_id': user_id}, {'_id': 0}) or {}

def save_portfolio(user_id: str, portfolio_dict: typing.Dict[str, typing.Any], db_name: typing.Optional[str] = None) -> None:
    collection = get_collection('portfolios', db_name)
    collection.update_one({'user_id': user_id}, {'$set': portfolio_dict}, upsert=True)

def fetch_price_history(symbol: str, db_name: typing.Optional[str] = None) -> typing.List[typing.Dict[str, typing.Any]]:
    collection = get_collection('price_history', db_name)
    return list(collection.find({'symbol': symbol}, {'_id': 0}).sort('date', 1))

def save_price_history(symbol: str, history: typing.List[typing.Dict[str, typing.Any]], db_name: typing.Optional[str] = None) -> None:
    collection = get_collection('price_history', db_name)
    for record in history:
        doc = {**record, 'symbol': symbol}
        key = {'symbol': symbol, 'date': doc.get('date')}
        collection.update_one(key, {'$set': doc}, upsert=True)

def test_connection():
    try:
        client = get_mongo_client()
        # Use configured default database
        db = client[settings.mongodb_db_name]
        collection = db['test_collection']
        test_doc = {'message': 'MongoDB connection successful!'}
        result = collection.insert_one(test_doc)
        logger.info(f"Inserted document with _id: {result.inserted_id}")
        doc = collection.find_one({'_id': result.inserted_id})
        logger.info(f'Fetched document: {doc}')
        collection.delete_one({'_id': result.inserted_id})
        logger.info('Test document deleted.')
    except DatabaseConnectionError as e:
        logger.critical(f'Database connection failed: {e}')

if __name__ == "__main__":
    test_connection()
