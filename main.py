from seam_carving import SeamCarver
import os

def image_resize_without_mask(input_image, output_image, new_height, new_width, insert_mode):
    obj = SeamCarver(input_image, new_height, new_width)
    if insert_mode:
        obj.resize()  # Seam insertion
    else:
        obj.resize()  # Seam removal
    obj.save_result(output_image)

def main():
    folder_in = 'in'
    folder_out = 'out'

    # Input filenames
    filename_input = 'smallim.jpg'
    filename_output = 'output_result.jpg'

    # Paths
    input_image_path = os.path.join(folder_in, "images", filename_input)
    output_image_path = os.path.join(folder_out, "images", filename_output)

    

    # Display options for the user
    print("Select the function to execute:")
    print("1. Resize image by seam removal")
    print("2. Resize image by seam insertion")

    choice = input("Enter your choice (1 or 2): ")
    
    # Prompt user for new height and width
    try:
        new_height = int(input("Enter the desired new height: "))
        new_width = int(input("Enter the desired new width: "))
    except ValueError:
        print("Invalid input! Please enter numeric values for height and width.")
        return

    try:
        if choice == '1':
            image_resize_without_mask(input_image_path, output_image_path, new_height, new_width, insert_mode=False)
            print(f"Resized image saved to {output_image_path} using seam removal.")
        elif choice == '2':
            image_resize_without_mask(input_image_path, output_image_path, new_height, new_width, insert_mode=True)
            print(f"Resized image saved to {output_image_path} using seam insertion.")
        else:
            print("Invalid choice. Please select 1 or 2.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
