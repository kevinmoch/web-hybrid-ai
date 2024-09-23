# Web API for Hybrid AI

Draft Proposal, September 2024

**This version:** [https://github.com/kevinmoch/web-hybrid-ai/](https://github.com/kevinmoch/web-hybrid-ai/blob/main/WebHybridAI.md)

**Issue Tracking:** [GitHub](https://github.com/kevinmoch/web-hybrid-ai/issues)

**Editors:** Chunhui Mo (Huawei), Martin Alvarez (Huawei)

**Translations:** [简体中文](https://github.com/kevinmoch/web-hybrid-ai/blob/main/README.zh-CN.md)

<hr>

## Abstract

This document describes Web APIs that allow web developers to directly access both on-device and cloud-based language models, and securely share user data between multiple apps when using these models.

## Status of this document

This specification was published by Huawei. It is not a W3C Standard nor is it on the W3C Standards Track.

## 1. Introduction

The following are the APIs goals:

- Provide web developers with a connection strategy for accessing both on-device and cloud-based models. For example, if no on-device models are available, attempt to access cloud-based models. Conversely, if cloud-based models are unavailable, try accessing on-device models.

- Provide web developers with a storage strategy for sharing user's private data. For example, one web app saves users' private data into a local vector database. Another web app, when accessing a on-device language model, can leverage this data through a local RAG system.

The following are not within our scope of concern:

- Design a uniform JavaScript API for accessing browser-provided language models, known as the [Prompt API](https://github.com/explainers-by-googlers/prompt-api), which is currently being explored by Chrome's built-in AI team.

- Issues faced by hybrid AI, such as model management, elasticity through hybrid AI, and user experience, as this topic has already been discussed in [Hybrid AI Presentations](https://github.com/webmachinelearning/hybrid-ai/tree/main/presentations) in the WebML IG, and will be covered in the sessions on [AI Model Management](https://github.com/w3c/tpac2024-breakouts/issues/15).

## 2. Connection API

### 2.1 Connection API Goals

Provide web developers with a connection strategy for accessing both on-device and cloud-based models.

### 2.2 Connection API Definition in Web IDL

```webidl
interface ModelConfig {
  DOMString? model;
  DOMString? baseUrl;
  // ...
};

enum ConnectionPreference { "remote", "local" };

interface ConnectConfig {
  // Cloud-based models
  record<DOMString, ModelConfig> remotes;

  // On-device models
  record<DOMString, ModelConfig>? locals;

  // Priority for accessing cloud-based models
  sequence<DOMString>? remote_priority;

  // Priority for accessing on-device models
  sequence<DOMString>? local_priority;

  // Models connection preference
  ConnectionPreference? prefer;
};

interface AIAssistant {
  Promise<AIAssistant> switchModel(DOMString modelName);
};

Promise<AIAssistant> connect(ConnectConfig connectConfig);
```

Some notes on this API:

- The `connect` method returns an `AIAssistant` that works just like the one in Chrome's `Prompt API`. For more details, check out [AIAssistant](https://github.com/explainers-by-googlers/prompt-api?tab=readme-ov-file#full-api-surface-in-web-idl). This API is considered an extension of the `AIAssistantFactory` in the `Prompt API`, with the `connect` method building on the `create` method to get around the limitation of using only one on-device model by default. The following is the Web IDL representation of the `connect` and `create` methods in the `AIAssistantFactory` of the `Prompt API`.

```webidl
[Exposed=(Window,Worker)]
interface AIAssistantFactory {
  Promise<AIAssistant> connect(ConnectConfig connectConfig);
  Promise<AIAssistant> create(optional AIAssistantCreateOptions options = {});
  // ...
};
```

- The `switchModel` method lets the `AIAssistant` switch between different LLMs. This API is also considered an extension of the `AIAssistant` in the `Prompt API`. Since the `connect` method's `connectConfig` parameter includes configurations for multiple models, once connect is successfully called, the returned `AIAssistant` can switch between these models by name. The following is the Web IDL representation of the `switchModel` method in the `AIAssistant` of the `Prompt API`.

```webidl
[Exposed=(Window,Worker)]
interface AIAssistant : EventTarget {
  Promise<AIAssistant> switchModel(DOMString modelName);
  Promise<DOMString> prompt(DOMString input, optional AIAssistantPromptOptions options = {});
  // ...
};
```

### 2.3 Connection API Sample Usages

```js
const config = {
  // Cloud-based models
  remotes: {
    gemini: {
      model: 'gemini-1.5-flash'
    }
  },
  // On-device models
  locals: {
    gemma: {
      randomSeed: 1,
      maxTokens: 1024
    },
    llama: {
      baseUrl: 'http://localhost:11434'
    },
    geminiNano: {
      temperature: 0.8,
      topK: 3
    }
  },
  // Priority for accessing on-device models
  local_priority: ['llama', 'gemma', 'geminiNano'],
  // Models connection preference
  prefer: 'remote'
}

// Connect to the remote Gemini model based on the above config
const session1 = await ai.connect(config)
const result1 = await session1.prompt('who are you')

// Switch to the Llama model that has already been defined in the config
const session2 = await session1.switchModel('llama')
const result2 = await session2.prompt('who are you')
```

Some notes on this sample:

- When you call the `connect` method to connect to a model, it follows the config settings. The preference is to use a `remote` model first, which in this case is the `gemini` model defined under `remotes`. If connecting to `gemini` fails, it will then try to connect to the models listed in `locals`, following the order in the `local_priority` array. So, it will attempt to connect to `llama` first, then `gemma`, and finally `geminiNano`.

- When you call the `switchModel` method to change models, it returns a new session for the switched model, while the original session remains unchanged and can still be used. This design allows you to access sessions for all the models defined in the config from the `connect` method. The advantage of this approach is that it enables web apps to assign different models to different levels of users.

- Before you use the `connect` and `switchModel` methods, you might need to set up a few things. For instance, you could need to apply for an API key to access the `gemini` model, download the `llama` and `gemma` models to your device, and maybe even run services like `ollama` and the `chroma` database on your local machine. But all of these tasks should be handled by browser vendors when they implement the APIs, not by web developers or users.

### 2.4 Connection API Implementation References

```js
// Define an asynchronous function to connect to the Gemini model
const gemini = async (options = {}) => {
  // Create a new instance of the GoogleGenerativeAI class using the API key
  const genAI = new GoogleGenerativeAI(gemini_api_key)

  // Define default options for the Gemini model (version 1.5-flash)
  const defaultOption = { model: 'gemini-1.5-flash' }

  // Get the generative model from GoogleGenerativeAI by merging default and custom options
  const model = genAI.getGenerativeModel({
    ...defaultOption, // Use the default Gemini model version
    ...options // Override or add any additional options passed in
  })

  // Define a function to generate a response from the model based on the input content
  const generateResponse = async (content, display) => {
    try {
      // Call the model's generateContentStream method to get a stream of generated content
      const result = await model.generateContentStream([content])

      // Stream the generated content in chunks and display it
      for await (const chunk of result.stream) {
        display(chunk.text()) // Display each chunk of text
      }
      display('', true) // Signal the end of the stream
    } catch (error) {
      throw error.message
    }
  }

  // Return the generateResponse function as part of the object
  return { generateResponse }
}

// Define an asynchronous function to connect to the Gemma model
const gemma = async (options = {}) => {
  // ...
}

// Define an asynchronous function to connect to the Llama model
const llama = async (options = {}) => {
  // ...
}

// Define an asynchronous function to connect to the GeminiNano model
const geminiNano = async (options = {}) => {
  // ...
}

// Add the Gemini model function to the models object, along with others like gemma, llama, and geminiNano
const models = { gemini, gemma, llama, geminiNano }

// Tries to connect to models based on a prioritized list
const tryConnect = async (prior) => {
  let model = null // Holds the connected model once successful
  let connect = null // Stores the function used to connect to the model

  // Loop through the prioritized list of models
  for (let i = 0; i < prior.length; i++) {
    // Get model name, connection method, and options
    const [name, connectModel, options] = prior[i]
    try {
      // Try to connect to the model
      model = await connectModel(options)
    } catch (error) {
      console.error('An error occurs when connecting the model', name, '\n', error)
    }
    if (model) {
      console.warn('Connect model', name, 'successfully')
      // Store the connect function
      connect = connectModel
      break // Exit the loop once connected to a model
    }
  }

  // Return the connected model and the connection function
  return [model, connect]
}

// Function to switch models dynamically
const switchModelFn = (prior, remotes, locals) => async (modelName) => {
  // Get the connection function for the given model
  const connectModel = models[modelName]
  // Get the configuration options from remotes or locals
  const options = remotes[modelName] || locals[modelName]
  // Connect to the new model
  const model = await connectModel(options)
  // Create a new session with the switched model
  return createSession(model, connectModel, prior, remotes, locals)
}

// Function that handles generating a prompt with the model
const promptFn = (model, connect, prior) => {
  return async (...args) => {
    try {
      // Try to generate a response using the current model
      return await model.generateResponse.apply(model, args)
    } catch (error) {
      console.error('Prompt failed when using the model\n', error)

      // If prompt fails, try switching models from the prioritized list
      for (let i = 0; i < prior.length; i++) {
        const [name, connectModel, options] = prior[i]
        // Only switch if the model is different
        if (connect !== connectModel) {
          try {
            // Try to connect to the alternate model
            const subModel = await connectModel(options)
            console.warn('Prompt failed, switch the model', name, 'successfully')
            // Retry the prompt with the new model
            return await subModel.generateResponse.apply(subModel, args)
          } catch (error) {
            console.error('Prompt failed, an error occurs when switching the model', name, '\n', error)
          }
        }
      }
    }
  }
}

// Creates a session with the connected model
const createSession = (model, connect, prior, remotes, locals) => {
  if (model) {
    return {
      // Provide a function to generate prompts
      prompt: promptFn(model, connect, prior),
      // Provide a function to switch models
      switchModel: switchModelFn(prior, remotes, locals)
    }
  } else {
    throw new Error('No available model can be connected!')
  }
}

// Connects to a model based on the provided configuration
const connect = async ({ remotes = {}, remote_priority, locals = {}, local_priority, prefer } = {}) => {
  // Get remote model names from priority list or default to all remotes
  const remoteNames = remote_priority || Object.keys(remotes)
  // Prepare an array of remote models
  const remote = remoteNames.map((name) => [name, models[name], remotes[name]])

  // Get local model names from priority list or default to all locals
  const localNames = local_priority || Object.keys(locals)
  // Prepare an array of local models
  const local = localNames.map((name) => [name, models[name], locals[name]])

  // Determine the priority order based on user preference (local or remote first)
  const prior = prefer === 'local' ? local.concat(remote) : remote.concat(local)
  // Try to connect to a model from the prioritized list
  const [model, connect] = await tryConnect(prior)

  // Create a session with the connected model
  return createSession(model, connect, prior, remotes, locals)
}
```

Explanation of major parts:

- **gemini function**: This function connects to the Google Gemini model (version gemini-1.5-flash by default) using the GoogleGenerativeAI class and an API key. It prepares a model with options and returns a generateResponse function.

- **generateResponse**: This function takes content (the input for the model) and a display callback to handle the output. It streams the model's response chunk by chunk, calling display for each chunk of text. If an error occurs during the generation, it throws the error.

- **models object**: This is where the gemini function is included alongside other models like gemma, llama, and geminiNano. These models can now be referenced in the broader script for connection and switching purposes.

- **tryConnect**: Iterates through a priority list of models, attempting to connect to each one until a successful connection is made.

- **switchModelFn**: Allows for switching to another model during runtime.

- **promptFn**: Sends a prompt to the model to generate a response. If the model fails, it attempts to switch to a different model and retry.

- **createSession**: Creates and returns a session for interacting with the model, including handling prompts and model switching.

- **connect**: Main function that connects to either a remote or local model, based on the priority order defined in the configuration.

You can find the source code for this implementation on GitHub at [Connect AI](https://github.com/kevinmoch/web-hybrid-ai/blob/main/ai.js).

## 3. Storage API

### 3.1 Storage API Goals

Provide web developers with a storage strategy for sharing user's private data.

### 3.2 Storage API Definition in Web IDL

```webidl
[Exposed=(Window,Worker)]
interface AIAssistantFactory {
  // Inserts a new entry and returns its entryId
  Promise<DOMString> insertEntry(DOMString category, DOMString content);

  // Updates an existing entry by its entryId
  Promise<boolean> updateEntry(DOMString entryId, DOMString content);

  // Removes an entry by its entryId
  Promise<boolean> removeEntry(DOMString entryId);

  Promise<AIAssistant> connect(ConnectConfig connectConfig);
  Promise<AIAssistant> create(optional AIAssistantCreateOptions options = {});
  // ...
};

[Exposed=(Window,Worker)]
interface AIAssistant : EventTarget {
  Promise<AIAssistant> switchModel(DOMString modelName);
  Promise<DOMString> prompt(DOMString input, optional AIAssistantPromptOptions options = {});
  // ...
};

dictionary AIAssistantPromptOptions {
  DOMString[] categories;
  // ...
};
```

Some notes on this API:

- Just like the `connect` method, the `insertEntry`, `updateEntry`, and `removeEntry` methods can also be considered extensions of the `AIAssistantFactory` in the `Prompt API`. The `insertEntry` method has a parameter called `category`, which indicates the category of the `entry`. When you call the `prompt` method, you can use the `categories` array in the options parameter, and this array can include one or more of these category names.

- The `entry` data type is a string, representing any kind of text, including a user's private data. This data will be used when the LLM performs inference. Usually, user information is categorized, such as identity data, interests, preferences, social network data, etc. This allows the LLM to reference only the relevant parts of the data, improving the speed of the inference process.

- It's important to note that collecting user data requires the user's explicit consent. So, before calling any of these APIs, a web app must get the user's approval. To ensure user privacy and data security, this data should only be stored locally on the user's device, such as in a shared local vector database. On-device models are the only one allowed to read this data during inference.

- Different web apps will store the user data they collect locally. Then, using this shared data, local LLMs can perform inference tailored to the user's needs, making the LLM's output more personalized. This is almost like the web apps are sharing user data indirectly. However, the actual implementation of these APIs will depend on browser vendors.

### 3.3 Storage API Sample Usages

```js
// Web App A connects to a cloud-based model
const remoteSession = await ai.connect(remoteConfig)

// Web App A fetches flight info based on the user’s travel plan
const flightInfo = await remoteSession.prompt(userPlan)

// Web App A stores the flight info in the user’s personalized data
await ai.insertEntry('travel', flightInfo)

// =====================================================

// Web App B connects to an on-device model
const localSession = await ai.connect(localConfig)

// Web App B stores the user's info into their personalized profile
await ai.insertEntry('travel', userInfo)

// Web App B uses the stored user data and flight info to suggest a list of hotels
const hotelList = await localSession.prompt(hotelDemand, { categories: ['travel'] })
```

Some notes on this sample:

- The first part of the code is from Web App A, which connects to a cloud-based model and retrieves flight info for a user based on their travel plans. Once it gets the flight info, the app stores that data as part of the user's personalized profile under a `travel` category.

- In the second part, Web App B connects to an on-device model. This app first collects some user-specific info and stores it under the `travel` category as well. Then, it uses both the stored user info and the flight data (from the previous step in Web App A) to suggest a personalized list of hotels for the user.

- In summary, Web App A gets flight info using a cloud-based model, and Web App B takes that info, along with other user data stored locally, to suggest hotels using an on-device model. Both apps are adding relevant data to the same personalized profile, making it easy for the user to get more customized suggestions as they move between different web apps.

### 3.4 Storage API Implementation References

```js
import { ChromaClient } from 'chromadb'
import { Chroma } from '@langchain/community/vectorstores/chroma'
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama'

// Define the collection name for the vector store, used to categorize and store entries
const collectionName = 'opentiny'

// Define Ollama settings, specifying the 'llama3:8b' model
const ollamaSetting = { model: 'llama3:8b' }

// Initialize Ollama embeddings with the specified model configuration
const embeddings = new OllamaEmbeddings(ollamaSetting)

// Function to insert a new entry into the Chroma vector store
const insertEntry = async (category, content) => {
  // Create a new Chroma vector store instance with the given embeddings and collection name
  const vectorStore = new Chroma(embeddings, { collectionName })

  // Add the document with the provided content and metadata (category)
  const ids = await vectorStore.addDocuments([
    {
      pageContent: content, // The actual content to be stored
      metadata: { category } // Metadata indicating the entry's category
    }
  ])

  // Return the ID of the newly added entry
  return ids[0]
}

// Function to update an existing entry in the Chroma vector store
const updateEntry = async (entryId, content) => {
  // Create a new Chroma vector store instance with the given embeddings and collection name
  const vectorStore = new Chroma(embeddings, { collectionName })

  // Update the document with the given content, matching it by entryId
  const ids = await vectorStore.addDocuments(
    [{ pageContent: content }], // The new content to update
    { ids: [entryId] } // ID of the entry to update
  )

  // Return true if the entry was successfully updated, otherwise false
  return ids[0] === entryId
}

// Function to remove an entry from the Chroma vector store
const removeEntry = async (entryId) => {
  // Create a new Chroma vector store instance with the given embeddings and collection name
  const vectorStore = new Chroma(embeddings, { collectionName })

  // Delete the document by its entry ID
  await vectorStore.delete({ ids: [entryId] })

  // Return true once the entry is successfully removed
  return true
}
```

Explanation of the implementation:

- The code demonstrates one way to store data locally, specifically using a vector database called `Chroma`. A vector database is built to store and retrieve high-dimensional vector data. `Chroma` is just one of these databases. In this case, we’re turning the text data (referred to as Entry) into high-dimensional vector representations and saving them locally. To do this, we first need to run the `Chroma` vector database locally and then use Chroma's API to store vectors.

- Once we've stored user behavior data, personal information, or any other content as vectors, we can then convert user queries or questions into vectors as well. These query vectors are used to search the vector database for the most similar entries, which serve as context. This context is then fed into a LLM, such as `Llama`, to generate more accurate and relevant responses.

- Earlier, in the `Connection API Implementation References`, we included a method for connecting to the Llama LLM. This following code shows how to connect `Chroma` and `Llama` to create a system that retrieves relevant context from a database and passes it to a LLM like `Llama` to generate answers. This process is made possible by using the `LangChain` framework for LLM app development and `Ollama`, which is a tool for deploying and running local LLMs.

You can find the source code for this implementation on GitHub at [Chroma Entry](https://github.com/kevinmoch/web-hybrid-ai/blob/main/chromaDB/entry.js).

```js
import { createRetrievalChain } from 'langchain/chains/retrieval'
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents'
import { PromptTemplate } from '@langchain/core/prompts'
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama'
import { Ollama } from '@langchain/community/llms/ollama'
import { Chroma } from '@langchain/community/vectorstores/chroma'

// Collection name for the Chroma database
const collectionName = 'opentiny'

// Define the prompt template that will be used by the LLM
const template = 'You are an expert, you are already aware of these: {context}, hence you can answer this question {input}'

// Settings for Ollama, using the Llama 3:8b model
const ollamaSetting = { model: 'llama3:8b' }

// Function to connect to Llama and generate responses
const llama = async (options = {}) => {
  // Initialize the LLM with Ollama, allowing for extra options
  const llm = new Ollama({ ...ollamaSetting, ...options })
  // Initialize embeddings to represent text as vectors
  const embeddings = new OllamaEmbeddings(ollamaSetting)

  // Set up the prompt template using the defined string
  const prompt = PromptTemplate.fromTemplate(template)
  // Create a chain that combines documents (context) and sends them to the LLM
  const combineDocsChain = await createStuffDocumentsChain({ llm, prompt })

  // Function to generate a response, taking input and category filters
  const generateResponse = async (input, display, category) => {
    try {
      // Retrieve relevant context from Chroma based on the category filter
      const vectorStore = await Chroma.fromExistingCollection(embeddings, { collectionName, filter: { category } })
      // Create a chain that retrieves relevant documents and sends them to the LLM
      const chain = await createRetrievalChain({
        retriever: vectorStore.asRetriever(), // Use Chroma as a retriever
        combineDocsChain // Combine the retrieved documents with the input query
      })

      // Stream the generated answer piece by piece
      const stream = await chain.stream({ input })
      for await (const chunk of stream) {
        if (chunk.answer) {
          display(chunk.answer) // Display the generated answer in real time
        }
      }
      display('', true) // Finish the stream
    } catch (error) {
      throw error.message // Handle any errors
    }
  }

  return { generateResponse } // Return the response generator
}
```

The technology used in the code is RAG, short for Retrieval Augmented Generation. It's a technique used in natural language processing (NLP) to improve the quality and relevance of generated text.

## 4 A Showcase of Hybrid AI App

### 4.1 Making Travel Plan

We’re planning a trip for the upcoming holiday. First, we need to open a flight booking web app to search for flights to our destination. The app has an AI assistant that uses a cloud-based model to recommend the best flights based on our personalized needs.

Next, we’ll open a hotel booking web app to find a place to stay based on our arrival date. This app also has an AI assistant, but it uses a on-device model to figure out our ideal hotel requirements based on personal info. Once it’s done, it sends that info to a cloud-based model to suggest the best hotel options for us.

### 4.2 Booking Flight & Hotel Architecture

![Booking App](WebHybridAI.png 'Booking Flight & Hotel Architecture')

As shown in the diagram above, the architecture of the flight booking app and the hotel booking app is as follows. The application process using Web API for Hybrid AI can be divided into the following steps:

1. The `Flight Assistant` in the flight booking app gathers the user's needs and sends a request to a remote `cloud-based model` to get flight info that matches the user’s preferences.

2. Once the flight info are retrieved, the `Flight Assistant` uses the `Storage API` to turn the flight data into vectorized `Categorized Entries`. These vectors are then stored in the `Chroma` database.

3. The `Hotel Assistant` in the hotel booking app collects user info, vectorizes it through the `Storage API`, and saves it as `Categorized Entries` in the `Chroma` database as well.

4. The `Hotel Assistant` then sends a request to a local `on-device model` to figure out the user’s hotel preferences, considering both the user info and flight info.

5. The `Hotel Assistant` then sends the hotel preferences to the `cloud-based model`, which returns a list of hotels that match the user’s specific needs.

This process ensures that both the flight and hotel suggestions are highly personalized by combining `on-device` and `cloud-based` AI models.

### 4.3 Booking Flight & Hotel Demo

The following demo simulates two independent apps interacting with local and remote AI models to recommend flights and hotels.

![Booking Demo](demo.gif 'Booking Flight & Hotel Demo')

As you can see in the animation, there are two web apps being demonstrated: a flight booking app on the left and a hotel booking app on the right. The demo process can be broken down into the following steps:

1. On the left, the flight booking app is accessed at http://127.0.0.1:5500. When you click the `Send Request` button, it sends a request to the remote `Gemini` model. Since LLMs don’t have real-time flight data, this demo just asks `Gemini` to predict a flight's arrival time.

2. The result from the `Gemini` model appears in blue text. By clicking `Insert Entry`, you can take part of that blue text (representing flight details) and store it in the `Chroma` vector database.

3. When you hit the `Query Entry` button, it shows the flight info you just saved in `Chroma` to confirm it’s correct. (Note: this button uses an API that’s not part of the `Storage API`, it’s just for demo purposes.)

4. On the right, the hotel booking app is accessed at http://localhost:3000, which uses a different port than the flight app (5500). So, it simulates two separate applications.

5. In the hotel booking app, clicking `Insert Entry` simulates storing the user's personal information in the `Chroma` vector database. Again, you can click `Query Entry` to check that the info was saved correctly.

6. Finally, when you click `Send Request` on the hotel booking app, it sends a request to the local `Llama` model, asking it to recommend a list of hotels based on the user’s personal info and flight destination.

7. The result from the `Llama` model shows up in blue text. It lists a hotel recommendation based on the user’s budget, destination, and other details, explaining why this hotel was chosen. Unlike the architecture described earlier, this demo skips the step of sending the request to a remote model, just to keep things simple.

You can find the source code for this demo on GitHub at [Flight Demo](https://github.com/kevinmoch/web-hybrid-ai/blob/main/chromaDB/flight.html) and [Hotel Demo](https://github.com/kevinmoch/web-hybrid-ai/blob/main/chromaDB/hotel.html).

### 4.4 Connection API Demo

In the previous section, `Connection API Sample Usages`, we learned that with the following connection configuration, we typically try to connect to the `Gemini` model first. If that fails, we then attempt to connect to `Llama`, `GeminiNano`, and `Gemma` models in that order.

```js
const config = {
  // Cloud-based models
  remotes: {
    gemini: {
      model: 'gemini-1.5-flash'
    }
  },
  // On-device models
  locals: {
    llama: {
      baseUrl: 'http://localhost:11434'
    },
    geminiNano: {
      temperature: 0.8,
      topK: 3
    },
    gemma: {
      randomSeed: 1,
      maxTokens: 1024
    }
  },
  // Priority for accessing on-device models
  local_priority: ['llama', 'geminiNano', 'gemma'],
  // Models connection preference
  prefer: 'remote'
}

// Connect to the remote Gemini model based on the above config
const session1 = await ai.connect(config)
const result1 = await session1.prompt('who are you')
```

The following animation shows an example of what happens when we call the `connect` method. It tries to connect to the `Gemini`, `Llama`, and `GeminiNano` models one by one, but they all fail. Finally, it successfully connects to the `Gemma` model. To make it easier to see what's happening in the browser console, we've added some messages to show when the connection is successful or fails.

![Connection Demo](connect.gif 'Connection API Demo')

You can find the source code for this demo on GitHub at [Unified Demo](https://github.com/kevinmoch/web-hybrid-ai/blob/main/webAI/unified.html).

## 5 Considerations for APIs

Several aspects of the Web API for Hybrid AI deserve our considerations.

### 5.1 Connection Strategy

In addition to the existing `Connection APIs`, here are some potential connection strategies that should be considered:

- **Connection Timeout Settings:**

  - Assign a connection timeout to each model individually, so that if a specific model takes too long to respond, it can be skipped or retried.
  - Alternatively, provide a global timeout setting that applies uniformly across all models, ensuring consistent behavior when connecting to multiple models.

- **Custom Connection Strategy:**

  - Expose a user-defined function that allows developers to create custom logic for deciding which model to access based on various conditions (e.g., user preferences, data size, model capabilities, etc.).
  - This gives developers more fine-grained control, enabling dynamic selection between `on-device` or `cloud-based` models depending on the scenario.

- **Model Status Information:**
  Provide developers with the ability to check the status of each configured model before sending a request. This includes information like:

  - Whether the model is available and can be accessed.
  - Whether the model is currently busy processing other requests.
  - Whether the model is active or idle, so that decisions can be made on whether to wait for a model or switch to an alternative one.

- **Load Balancing Across Models:**
  Introduce load-balancing mechanisms that automatically distribute requests across multiple models based on their current load, availability, and performance, ensuring smoother operation in busy systems.

- **Model Version Control and Compatibility Checks:**
  Ensure the system can handle different versions of models, check for compatibility, and automatically switch to the latest or most compatible version as needed.

### 5.2 Storage Strategy

In addition to the existing `Storage APIs` that integrate with a local `Chroma` vector database, several other storage strategies should be considered to ensure flexibility, efficiency, and reliability. Here are some potential storage strategies:

- **Leverage Existing Browser Storage:**
  Utilize browser-native storage mechanisms like `localStorage` or `IndexedDB` to store entry information on the user's local device. This allows lightweight storage of user-specific data without needing a full vector database.

- **Configurable Multiple Local Vector Databases:**
  Similar to Connection APIs, provide developers the option to configure multiple local vector databases. Each LLM could then be paired with a specific vector store, depending on the use case or model requirements.

- **Capacity Management:**
  When storing data as public or shared entries, consider storage limitations, especially in environments with restricted space (e.g., browser-based storage or low-resource devices).

- **Error Handling for Database Failures:**
  Automatically retry failed storage operations or switch to an alternative storage method. If database corruption occurs, notify users or developers and provide recovery options such as automatic backups or rollback to a previous stable state.

- **Support for Different Vector Storage Formats:**
  Provide flexibility to developers by allowing the storage of vectors in different formats, such as JSON (for simple structures) or binary formats (for efficient storage and retrieval of large vector datasets).

### 5.3 Native OS APIs

If `Connection APIs` and `Storage APIs` were natively supported at the OS level (or if the browser was simply calling these underlying native OS APIs), it would bring several benefits:

- **Cross-App Data Sharing on Local Devices:**
  Whether it's a web app running in a browser or a native app on a device (such as an iOS app, Android app, or Windows app), all these apps could share the user data they've collected and store it locally on the device. This means that any app on the device—be it a web app or native app—can leverage this shared information when using local models for inference. As a result, the models' responses will be more personalized and aligned with the user's needs.

- **Seamless Experience Across Devices:**
  Suppose a user has multiple devices running the same OS or from the same manufacturer, like HarmonyOS or MacOS & iOS. When the user logs into these devices, the OS could recognize all the devices under that user’s account. With the user’s consent, the OS could securely share data across the devices through a safe transmission channel. This would ensure that when different local models on each device perform inference, they can produce results that are consistent with the user's personalized preferences, no matter which device is being used.

- **Improved Privacy and Security:**
  Since data sharing would happen at the OS level, it would benefit from the OS’s built-in security protocols. For instance, end-to-end encryption or secure enclaves could be used to ensure that only authorized apps and models have access to the data, reducing privacy risks while still allowing for seamless cross-app functionality.

- **Unified Data Management:**
  Users wouldn’t need to manually move their data between apps or platforms. The OS could manage data centrally, allowing apps to access the most relevant, up-to-date information, resulting in more coherent and personalized model responses across different devices and apps.

- **Better Performance:**
  Using native OS-level APIs could lead to faster and more efficient data storage and retrieval processes. By eliminating the overhead of browser-based storage mechanisms, the system could store larger volumes of data and handle more complex operations, especially when working with vector databases or large AI models.
