docker run -d \
  --network host \
  --name mytts-frontend \
  -v ${PWD}/frontend:/app \
  -p 7724:7724 \
  -w /app \
  iss/streamlit:v1.0 \
  streamlit run app.py --server.address 0.0.0.0 --server.port 7724