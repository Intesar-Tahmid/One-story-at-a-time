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
