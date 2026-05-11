# agents-orchestration


## Running Neo4j

1. make sure you install all libs in `requirements.txt`
2. update your `.env` to include Neo4j vars
3. run `docker compose up -d`
4. Load the mock data by running `python -m scripts.load_mock_neo4j`

Now you can visit `http://localhost:7474`