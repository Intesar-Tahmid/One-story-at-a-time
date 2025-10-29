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
