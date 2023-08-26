import os
import secrets

# Using os.urandom()
key = os.urandom(16)
print(f"key using os: {key}")

# Using secrets.token_bytes()
key = secrets.token_bytes(16)
print(f"key using secrets: {key}")

# Using secrets.token_hex() for a hex-format key
key = secrets.token_hex(16)
print(f"key using secrets as hex: {key}")

