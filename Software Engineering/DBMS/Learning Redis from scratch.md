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
``