
1. Generate text (as before):
   ```
   POST http://localhost:3030/generate
   Content-Type: application/json
   
   {
     "prompt": "Once upon a time"
   }
   ```

2. Train the model on a Wikipedia topic:
   ```
   POST http://localhost:3030/train
   Content-Type: application/json
   
   {
     "topic": "Artificial_intelligence"
   }
   ```

This will fetch the Wikipedia article on Artificial Intelligence and use it to train the Markov model.
