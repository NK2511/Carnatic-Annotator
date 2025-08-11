import os
import shutil
import subprocess

def process_folder_with_audio(input_folder):
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    output_folder = os.path.join(downloads_folder, "Carnatic_Vocals")
    os.makedirs(output_folder, exist_ok=True)

    supported_formats = [".wav", ".mp3"]

    for filename in os.listdir(input_folder):
        if not any(filename.lower().endswith(ext) for ext in supported_formats):
            continue

        input_file_path = os.path.join(input_folder, filename)
        file_stem = os.path.splitext(filename)[0]
        final_vocal_path = os.path.join(output_folder, f"{file_stem}_vocals.wav")

        if os.path.exists(final_vocal_path):
            print(f"🟡 Skipping {filename} — vocals already extracted.")
            continue

        print(f"🔧 Extracting vocals from: {filename}...")

        # Use spleeter with 2stems (vocals + accompaniment)
        temp_output = os.path.join(input_folder, "temp_spleeter_output")
        command = [
            'spleeter', 'separate',
            '-i', input_file_path,
            '-p', 'spleeter:2stems',
            '-o', temp_output
        ]

        try:
            subprocess.run(command, check=True)

            # Move just vocals.wav to output folder and rename
            source_vocal = os.path.join(temp_output, file_stem, "vocals.wav")
            shutil.move(source_vocal, final_vocal_path)
            print(f"✅ Saved vocals to: {final_vocal_path}")

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {filename} — {e}")

        # Clean up temp folders
        shutil.rmtree(temp_output, ignore_errors=True)

if __name__ == "__main__":
    folder_with_audio = r"C:\Users\nandh\Downloads\Carnatic _music"  # your folder
    process_folder_with_audio(folder_with_audio)
