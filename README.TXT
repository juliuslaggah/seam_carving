# Seam Carving Image Resizer

This project implements a seam carving algorithm to resize images dynamically while preserving important visual content. Users can resize images either by seam removal or insertion. The program is implemented in Python and leverages libraries such as NumPy, PIL, and SciPy.

---

## Prerequisites

1. **Python Installation**:
   - Ensure Python 3.6 or later is installed.

2. **Required Libraries**:
   Install the required Python libraries by running:
   ```bash
   pip install numpy pillow scipy
   ```

---

## Project Structure

- `main.py`: Entry point for the application.
- `seam_carving.py`: Contains the SeamCarver class implementing the core seam carving algorithm.
- `in/images`: Folder containing input images.
- `out/images`: Folder where output images will be saved.

---

## Setting Up

1. download the project files.
2. Navigate to the project directory in the terminal.
3. Place the image you want to resize into the `in/images` directory.

---

## Running the Program

1. **Launch the Script**:
   ```bash
   python main.py
   ```

2. **Follow the Prompts**:
   - The terminal will display options:
     ```
     Select the function to execute:
     1. Resize image by seam removal
     2. Resize image by seam insertion
     ```
   - Enter `1` to resize the image by removing seams.
   - Enter `2` to resize the image by inserting seams.

3. **Input Desired Dimensions**:
   - After selecting an option, the program will prompt you to enter the new height and width:
     ```
     Enter the desired new height:
     Enter the desired new width:
     ```

4. **Output**:
   - The resized image will be saved to the `out/images` directory with the name `output_result.jpg`.
   - The terminal will confirm:
     ```
     Resized image saved to out/images/output_result.jpg
     ```

---

## Example Workflow

1. Place `example.jpg` in `in/images`.
2. Run the program:
   ```bash
   python main.py
   ```
3. Select an option (e.g., `1` for seam removal).
4. Enter the new height and width (e.g., `500` and `700`).
5. The resized image will appear in `out/images/output_result.jpg`.

---

## Notes

- Ensure the input image file exists in the `in/images` folder.
- Output dimensions should not exceed reasonable limits based on the original image size.

---

