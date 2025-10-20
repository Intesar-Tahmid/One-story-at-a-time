Storing data is only interesting if you intend to retrieve that data later and use according to your needs.
- Data Models describe the data
- Database specify how the data should be stored
We'll learn about two distinct types of data:
- Historical data in data storage engines
- Streaming data in real-time transports

## Data Sources
One very common source is user input data. The issue is it can easily be mal-formatted. If input is supposed to be texts, they might be too long or too short. If it's supposed to be numerical values, users might accidentally enter texts.
Another source is system generated data. They might come from logs or system outputs. Logs are less likely to be malformatted. 😼
Since debugging ML systems is pretty hard it's a very good practice to log everything possible. This approach brings two problems:
- It can be hard to know where to look, because of huge logs
- How to store rapidly growing logs
In most cases you have to store logs as long as they're useful. If you don't have to access your logs frequently, they can can be stored in low-access storage that costs less than higher frequency access storage.

- First Party Data - Company already collects about your users or customers
- Second Party Data - Data collected by another company on their own customers that they make available to you
- Third party Data - Companies collect data on the public who aren't their customers.



### Higher Frequency access storage & Lower frequency access storage
**Common Technologies for higher frequency access:**

- **CPU Cache (L1, L2, L3):** The absolute fastest memory, located directly on the processor chip.
    
- **RAM (Random Access Memory):** The computer's main memory. Data must be loaded into RAM for the CPU to process it.
    
- **NVMe SSDs (Non-Volatile Memory Express Solid-State Drives):** The fastest type of storage drive available for persistent data (data that remains after power is off). Used for high-performance databases and applications.
    
- **SATA/SAS SSDs:** Faster than hard drives but slower than NVMe. A good balance of performance and cost for many applications.

**Common Technologies for lower frequency access:**

- **Hard Disk Drives (HDDs):** Especially large-capacity drives in massive arrays. Much slower than SSDs but much cheaper for bulk storage.
    
- **Magnetic Tape Storage:** An older technology that is still the cheapest and most durable option for massive-scale archival. Retrieval is very slow and sequential.
    
- **Cloud Archival Tiers (e.g., AWS Glacier, Azure Archive Blob Storage):** These services use a combination of technologies (likely including tape) and offer incredibly low prices. The trade-off is that retrieving data often involves a "retrieval time" of several minutes to several hours and may incur a cost.

## Data Formats
