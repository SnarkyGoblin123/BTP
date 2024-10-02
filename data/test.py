import struct

def read_iq_data(file_path):
    try:
        with open(file_path, 'rb') as file:
            # Read the entire file
            data = file.read()
            
            # Check if the data length is a multiple of 4
            if len(data) % 4 != 0:
                raise ValueError("The file size is not a multiple of 4 bytes, which is required for 16-bit I and Q data.")
            
            # Calculate the number of 16-bit integers
            num_samples = len(data) // 4  # Each sample consists of 2 16-bit integers (I and Q)
            
            # Unpack the binary data into 16-bit integers
            iq_data = struct.unpack(f'<{num_samples * 2}h', data)
            
            # Separate I and Q components
            I_data = iq_data[0::2]
            Q_data = iq_data[1::2]
            
            # Print the I and Q data
            print("I data:", I_data)
            print("Q data:", Q_data)
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
read_iq_data('fl.dat')