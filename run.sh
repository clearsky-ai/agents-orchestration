


python -m scripts.load_mock_neo4j --clear
rm /Users/salemjr/work/clearsky/agents-again/src/data/runtime_state.db
python -m scripts.replay_events --events src/data/scenario1_events.json