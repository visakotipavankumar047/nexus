"""Create the Pinecone index (PINECONE_INDEX) sized for the configured embedding model.

    python scripts/create_index.py [--cloud aws] [--region us-east-1]

If the index exists, only checks that its dimension matches the embedding model.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinecone import Pinecone, ServerlessSpec  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.llm import embedding_model_id, get_embeddings  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cloud", default="aws")
    p.add_argument("--region", default="us-east-1")
    args = p.parse_args()

    s = get_settings()
    dim = len(get_embeddings().embed_query("dimension probe"))  # truth comes from the model, not a constant
    pc = Pinecone(api_key=s.pinecone_api_key.get_secret_value())

    if pc.has_index(s.pinecone_index):
        desc = pc.describe_index(s.pinecone_index)
        ok = desc.dimension == dim
        print(f"index '{s.pinecone_index}' exists: dimension={desc.dimension} metric={desc.metric} "
              f"-> {'OK' if ok else f'MISMATCH, {embedding_model_id()} produces {dim}'}")
        sys.exit(0 if ok else 1)

    pc.create_index(name=s.pinecone_index, dimension=dim, metric="cosine",
                    spec=ServerlessSpec(cloud=args.cloud, region=args.region))
    print(f"created index '{s.pinecone_index}' dimension={dim} metric=cosine ({args.cloud}/{args.region})")


if __name__ == "__main__":
    main()
