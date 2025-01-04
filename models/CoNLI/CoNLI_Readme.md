To install necessary dependencies, see the instructions in the Semantic Entropy directory.

To start the API run the flask script in
```coNLI_api_script.sh```
It will likely run in port 5000, so you can access the API at http://localhost:5000/

The API uses OpenAI API so you need to have an API key. You can get one at https://beta.openai.com/signup/
It also requires an Azure Language key, which can be obtained at https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/

The API has one endpoint, /api/conli, which receives a GET request with a JSON body containing the prompt that you want to run. The output is then analyzed using semantic entropy, which will be outputted. The JSON body should have the following format:
```
{
  "id": placeholder,
  "api_key": "your_openai_key",
  "prompt": "The text you want to analyze"
  "inference_temperature": 1.0,
  "model": "gpt-4"
}
```

Some things that can be modified/customized:
- The model used for the analysis. The default is GPT-4, but you can change it to any of the models available in OpenAI.
- Entity detector. The number of entities to detect can be modified, as well as the type of entities to detect.
- Sentence selector. The criteria for analyzing a sentence for the analysis can be modified. The default is to analyze all sentences, but you can change it to only analyze sentences that contain a certain keyword, for example.
- NLI Batch size. For efficient prompting of GPT, chain of thought prompts are batched together. The batch size can be modified.
- Entity Batch size. For efficient entity detection, threads are parallelized. The batch size can be modified.
- Inference temperature. The temperature of the model can be modified. The default is 1.0, but you can change it to any value between 0 and 1.

The API is found in the file ```semantic_uncertainty/CoNLI_api.py```.