import pandas as pd

'''Reorder each line of the original csv file so that the even indexed values are in the front and the odd indexed values are in the back, 
and then save it to the new csv file, i.e., the first 100 values are x-coordinates, and the last 100 values are y-coordinates'''

def reorder_csv(file_path, output_path):
    # Read CSV file with 200 space-separated values per line
    data = pd.read_csv(file_path, header=None, sep='\\s+', engine='python')

    # Reorder the values in each row
    def reorder_row(row):
        even_indices = row[::2]  # The value of the even index
        odd_indices = row[1::2]  # Value of odd index
        return pd.Series(list(even_indices) + list(odd_indices))

    # Apply the reorder function to each row
    reordered_data = data.apply(reorder_row, axis=1)

    # Save the re-sorted data to a new CSV file
    reordered_data.to_csv(output_path, header=False, index=False, sep=' ')
if __name__ == '__main__':
    # Input file path
    input_file = 'pedestrians_positions_MI.csv'
    # Output File Path
    output_file = 'pedestrians_positions_MI_converted.csv'

    # Calling a function to reorder a CSV file
    reorder_csv(input_file, output_file)
