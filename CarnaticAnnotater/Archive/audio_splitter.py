import os
import subprocess

def split_single_audio_file(input_file_path):
    # Check if the file exists
    if not os.path.isfile(input_file_path):
        print(f"Error: The file {input_file_path} does not exist.")
        return

    # Create output directory
    output_directory = os.path.join(os.path.dirname(input_file_path), "output")
    os.makedirs(output_directory, exist_ok=True)

    try:
        # Construct the Spleeter command
        command = [
            'spleeter', 'separate',
            '-i', input_file_path,
            '-p', 'spleeter:4stems',
            '-o', output_directory
        ]

        # Run the command
        subprocess.run(command, check=True)
        print(f"✅ Successfully processed: {input_file_path}")
        print(f"🟢 Output saved in: {output_directory}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {input_file_path}: {e}")

if __name__ == "__main__":
    # Replace with your audio file path
    input_file_path = r"C:\Users\nandh\Downloads\carnatic_varnam_1.1\carnatic_varnam_1.1\Audio\223580__gopalkoduri__carnatic-varnam-by-ramakrishnamurthy-in-abhogi-raaga.mp3"
    split_single_audio_file(input_file_path)
