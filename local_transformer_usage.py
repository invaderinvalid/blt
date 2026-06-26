from blt.byte_encoder import UTF8Encoder
from blt.pipeline import BytePatchEncoder

text = '''life is a journey, not a destination. It is about the experiences we have, the lessons we learn, and the memories we create along the way. Embrace every moment, cherish the people you meet, and never stop growing.
'''

bytes_input = UTF8Encoder.encode(
    text
)

model = BytePatchEncoder()

patches, boundaries = model(
    bytes_input
)

print(f"Patches shape: {patches.shape}, Boundaries count: {len(boundaries)}")