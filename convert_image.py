import cairosvg
import glob
import os

for file in glob.glob("*.svg"):
    output = os.path.splitext(file)[0] + ".png"

    cairosvg.svg2png(
        url=file,
        write_to=output,
        output_width=1200
    )

    print(f"Converted: {file} -> {output}")