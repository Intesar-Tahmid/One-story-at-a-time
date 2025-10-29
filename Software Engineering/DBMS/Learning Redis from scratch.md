```
def __init__(self, host: str = 'localhost', port: int = 6379, 
             db: int = 0, password: str = None):
    """Initialize database connection"""
    self.connection = RedisConnection()
    self.connection.connect(host, port, db, password)
    
    self.loader = DataLoader()
    self.query = QueryEngine()
    self.analytics = Analytics()
```
- Redis supports 16 databases by default
- Above code is creating separate modules for different operations
### How to add a simple health check?
```
def is_connected(self) -> bool;
	return self.connection.ping()
```
### If there are multiple DBs, how to choose one?
```
def select_database(self, db:int)
	self.connection.select_db(db)
```
### How to add configuration validation?
```
def __init__(self, host: str = 'localhost', port: int = 6379, 
             db: int = 0, password: str = None):
    if not 0 <= db <= 15:
        raise ValueError("Redis database must be between 0 and 15")
    if port <= 0 or port > 65535:
        raise ValueError("Invalid port number")
        
```
### Connection Pooling
A technique where instead of creating a new database connection for every operation, you maintain a pool of pre-established connections that can be reused.
#### Without connection pooling
```
# BAD: New connection for each operation
def save_deal(deal_data):
    redis = redis.Redis(host='localhost', port=6379)  # NEW CONNECTION
    redis.hset(f"deal:{deal_data['id']}", mapping=deal_data)
    redis.close()  # CLOSE CONNECTION

# For 1600 deals: 1600 connections created/destroyed
```
#### With connection pooling
```
# GOOD: Reuse connections from pool
pool = redis.ConnectionPool(max_connections=10)
redis_client = redis.Redis(connection_pool=pool)

def save_deal(deal_data):
    # Reuses existing connection from pool
    redis_client.hset(f"deal:{deal_data['id']}", mapping=deal_data)

# For 1600 deals: Only 10 connections max, reused efficiently
```
- Without pooling it will take 1-5 ms to establish each new TCP connection
- With pooling 0.1ms to get connection from pool
- 