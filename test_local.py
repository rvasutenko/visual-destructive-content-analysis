from pipeline import DestructiveContentPipeline
import config

pipeline = DestructiveContentPipeline(config)

with open("landscape.png", "rb") as f:
    img_bytes = f.read()

result = pipeline.process_single_image(
    image_bytes=img_bytes,
    post_text="Умри, мразь!",
    group_name="название сообщества"
)

print(result)