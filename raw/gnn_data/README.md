# Main Data Folder

In here you want to place the raw ASOS data in csv format.

The directories list of files should be something like:

- `variant_table_training.csv`
- `variant_table_testing.csv`
- `customer_nodes_training.csv`
- `customer_nodes_testing.csv`
- `event_table_training.csv`
- `event_table_testing.csv`
- `links_constructor.py`

Once this is done, run the command `python links_constructor.py` which will create all the extra links you need for either the virual nodes or direct links for countries, brands, product types etc. You will only need to do this once!

After this you should be set up to use the rest of the code with no issues and the directory should look like:

- `variant_table_training.csv`
- `variant_table_testing.csv`
- `customer_nodes_training.csv`    
- `customer_nodes_testing.csv`     
- `event_table_training.csv`       
- `event_table_testing.csv`        
- `product_nodes_training.csv`
- `product_nodes_testing.csv`
- `brand_table_training.csv`       
- `brand_table_testing.csv`        
- `country_table_training.csv`     
- `country_table_testing.csv`      
- `productType_table_training.csv` 
- `productType_table_testing.csv`  
- `links_constructor.py`           
