from PIL import Image

# RGBA：R,G,B,Alpha；Alpha=0 为完全透明，255 为完全不透明
img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
img.save("test.png")
