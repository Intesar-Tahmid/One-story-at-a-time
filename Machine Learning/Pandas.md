**

# Basics:

- .head() returns the first few rows (the “head” of the DataFrame).
    
- .info() shows information on each of the columns, such as the data type and number of missing values.
    
- .shape returns the number of rows and columns of the DataFrame.
    
- .describe() calculates a few summary statistics for each column.
    

  

Dataframe objects consist of three components:

1. .values - 2D numpy array of values
    
2. .columns - Index of columns; the column names
    
3. .index - index of rows, either row numbers or row names
    

# Sorting and subsetting:

You can sort rows using this:
```python
dogs.sort_values("weight_kg")  
#dogs is a dataframe  
#It will sort the data according to weight from low to high

#or dogs.sort_values(“weight_kg”, ascending=False) if 

#you want to sort in a descending order
```


---
```python
is_lab = dogs["breed] == "Labrador"  
is_brown = dogs["color"] == "Brown"  
dogs[is_lab & is_brown]
```
