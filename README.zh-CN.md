# 混合 AI 的 Web API

草案提案，2024 年 9 月

**此版本:** [https://github.com/kevinmoch/web-hybrid-ai/](https://github.com/kevinmoch/web-hybrid-ai/blob/main/WebHybridAI.md)

**问题跟踪:** [GitHub](https://github.com/kevinmoch/web-hybrid-ai/issues)

**作者:** Chunhui Mo (Huawei), Martin Alvarez (Huawei)

**翻译:** [英文](https://github.com/kevinmoch/web-hybrid-ai/blob/main/README.md)

<hr>

## 摘要

本文档描述了 Web API，允许 Web 开发人员直接访问设备端和云端的语言模型，并在使用这些模型时安全地在多个应用之间共享用户数据。

## 文档状态

本规范由 Huawei 发布。它不是 W3C 标准，也不在 W3C 标准流程上。

## 1. 引言

以下是这些 API 的目标：

- 为 Web 开发人员提供访问设备端和云端模型的连接策略。例如，如果设备端没有可用模型，尝试访问云端模型；相反，如果云端模型不可用，尝试访问设备端模型。

- 为 Web 开发人员提供共享用户私密数据的存储策略。例如，一个 Web 应用将用户私密数据保存到本地向量数据库中，另一个 Web 应用在访问设备端语言模型时，可以通过本地 RAG（检索增强生成）系统利用这些数据。

以下不在我们的关注范围内：

- 设计一个统一的 JavaScript API，用于访问浏览器提供的语言模型，称为 [Prompt API](https://github.com/explainers-by-googlers/prompt-api)，目前由 Chrome 内置的 AI 团队探索。

- 混合 AI 面临的问题，如模型管理、通过混合 AI 实现弹性及用户体验等，已在 WebML IG 的 [Hybrid AI Presentations](https://github.com/webmachinelearning/hybrid-ai/tree/main/presentations) 中讨论，并将在 [AI Model Management](https://github.com/w3c/tpac2024-breakouts/issues/15) 的会议中涉及。

## 2. 连接 API

### 2.1 连接 API 的目标

为 Web 开发人员提供访问设备端和云端模型的连接策略。

### 2.2 Web IDL 中的连接 API 定义

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

关于此 API 的一些说明：

- connect 方法返回一个 AIAssistant，其功能与 Chrome 的 Prompt API 中的 AIAssistant 相同。更多详情请参考 [AIAssistant](https://github.com/explainers-by-googlers/prompt-api?tab=readme-ov-file#full-api-surface-in-web-idl)。此 API 被视为 Prompt API 中的 AIAssistantFactory 的扩展，其中 connect 方法基于 create 方法构建，以解决默认只能使用一个设备端模型的限制。以下是 Web IDL 表示的 connect 和 create 方法在 Prompt API 的 AIAssistantFactory 中的定义。

```webidl
[Exposed=(Window,Worker)]
interface AIAssistantFactory {
  Promise<AIAssistant> connect(ConnectConfig connectConfig);
  Promise<AIAssistant> create(optional AIAssistantCreateOptions options = {});
  // ...
};
```

- switchModel 方法允许 AIAssistant 在不同的 LLM（大语言模型）之间切换。这个 API 也被视为 Prompt API 中 AIAssistant 的扩展。由于 connect 方法的 connectConfig 参数包含多个模型的配置，一旦连接成功，返回的 AIAssistant 可以通过名称在这些模型之间切换。以下是 Prompt API 中的 AIAssistant 的 switchModel 方法的 Web IDL 表示。

```webidl
[Exposed=(Window,Worker)]
interface AIAssistant : EventTarget {
  Promise<AIAssistant> switchModel(DOMString modelName);
  Promise<DOMString> prompt(DOMString input, optional AIAssistantPromptOptions options = {});
  // ...
};
```

### 2.3 连接 API 示例用法

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

一些关于此示例的说明：

- 当调用 connect 方法连接到模型时，它会遵循配置设置。首选项是优先使用 remote 模型，即配置中的 gemini 模型。如果连接 gemini 失败，则会尝试按 local_priority 数组中的顺序连接本地模型，依次尝试连接 llama、gemma 和 geminiNano。

- 调用 switchModel 方法切换模型时，它会返回切换后的新会话，而原始会话保持不变，仍可使用。这样设计的好处是可以让 Web 应用为不同级别的用户分配不同的模型。

- 在使用 connect 和 switchModel 方法之前，可能需要设置一些前提条件。例如，可能需要申请访问 gemini 模型的 API 密钥，下载 llama 和 gemma 模型到本地设备，甚至需要在本地运行 ollama 和 chroma 数据库等服务。但所有这些任务应由浏览器供应商在实现 API 时处理，而不是由 Web 开发人员或用户处理。

### 2.4 连接 API 实现参考

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

主要部分的解释：

- **gemini 函数**: 此函数使用 GoogleGenerativeAI 类和 API 密钥连接到 Google Gemini 模型（默认版本为 gemini-1.5-flash），准备好模型并返回一个 generateResponse 函数。

- **generateResponse**: 此函数接受内容（模型的输入）和处理输出的显示回调函数。它逐块流式传输模型的响应，为每块文本调用一次显示。如果生成过程中发生错误，它将抛出错误。

- **models 对象**: gemini 函数与其他模型如 gemma、llama 和 geminiNano 一起包含在此对象中，这些模型可在更广泛的脚本中用于连接和切换。

- **tryConnect**: 通过模型的优先级列表进行迭代，尝试连接每个模型，直到成功连接为止。

- **switchModelFn**: 允许在运行时切换到另一个模型。

- **promptFn**: 向模型发送提示以生成响应。如果模型失败，它会尝试切换到另一个模型并重试。

- **createSession**: 创建并返回一个用于与模型交互的会话，包括处理提示和模型切换。

- **connect**: 主函数，根据配置中的优先顺序连接到远程或本地模型。

在 GitHub 上该实现的源代码：[Connect AI](https://github.com/kevinmoch/web-hybrid-ai/blob/main/ai.js).

## 3. 存储 API

### 3.1 存储 API 的目标

为 Web 开发人员提供共享用户私有数据的存储策略。

### 3.2 Web IDL 中的存储 API 定义

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

关于该 API 的一些说明：

- 类似于 connect 方法，insertEntry、updateEntry 和 removeEntry 方法也可以视为 Prompt API 中 AIAssistantFactory 的扩展。insertEntry 方法有一个名为 category 的参数，表示 entry 的类别。当调用 prompt 方法时，可以在 options 参数中使用 categories 数组，该数组可以包含一个或多个类别名称。

- entry 数据类型是一个字符串，表示任何类型的文本，包括用户的私有数据。当 LLM 进行推理时，会使用这些数据。通常，用户信息会进行分类，如身份数据、兴趣、偏好、社交网络数据等。这使得 LLM 仅引用相关部分数据，提升推理过程的速度。

- 需要注意的是，收集用户数据需要用户的明确同意。因此，在调用任何这些 API 之前，web 应用必须获得用户的批准。为了确保用户隐私和数据安全，这些数据应仅存储在用户设备上的本地，如共享的本地向量数据库。在推理过程中，仅允许本地模型读取这些数据。

- 不同的 web 应用程序将收集的用户数据本地存储。然后，使用这些共享的数据，本地 LLM 可以根据用户需求进行推理，使 LLM 的输出更加个性化。这几乎类似于 web 应用间间接共享用户数据。然而，这些 API 的具体实现取决于浏览器供应商。

### 3.3 存储 API 示例用法

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

关于该示例的一些说明：

- 代码的第一部分来自 Web App A，该应用程序连接到基于云的模型，并根据用户的旅行计划检索航班信息。一旦获取了航班信息，该应用程序将数据存储为用户个性化资料的一部分，归类于 travel 类别。

- 第二部分中，Web App B 连接到本地设备模型。该应用程序首先收集一些特定于用户的信息，并将其同样存储在 travel 类别中。然后，它使用存储的用户信息和航班数据（来自 Web App A 的上一步）为用户推荐个性化的酒店列表。

- 总的来说，Web App A 使用基于云的模型获取航班信息，Web App B 使用这些信息以及本地存储的其他用户数据，通过本地设备模型推荐酒店。两个应用程序都向同一个个性化资料添加了相关数据，使用户在不同的 web 应用间获得更定制化的建议。

### 3.4 存储 API 实现参考

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

实现的解释：

- 该代码演示了一种本地存储数据的方式，特别是使用一个名为 Chroma 的向量数据库。向量数据库用于存储和检索高维向量数据。Chroma 只是其中一种数据库。在这种情况下，我们将文本数据（称为 Entry）转换为高维向量表示，并将其本地存储。为此，我们首先需要在本地运行 Chroma 向量数据库，然后使用 Chroma 的 API 来存储向量。

- 一旦我们将用户行为数据、个人信息或其他内容存储为向量，我们还可以将用户查询或问题转换为向量。这些查询向量用于在向量数据库中搜索最相似的条目，作为上下文。然后将这些上下文提供给 LLM，例如 Llama，以生成更准确和相关的响应。

- 在之前的 Connection API 实现参考 中，我们包含了连接 Llama LLM 的方法。下面的代码展示了如何连接 Chroma 和 Llama，以创建一个从数据库检索相关上下文并传递给 LLM（如 Llama）生成答案的系统。通过使用 LangChain 框架开发 LLM 应用程序和 Ollama（一种用于本地 LLM 部署和运行的工具），可以实现这一过程。

在 GitHub 上该实现的源代码： [Chroma Entry](https://github.com/kevinmoch/web-hybrid-ai/blob/main/chromaDB/entry.js).

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

## 4 混合 AI 应用展示

### 4.1 制定旅行计划

我们正在为即将到来的假期计划一次旅行。首先，我们需要打开一个航班预订 web 应用，搜索飞往目的地的航班。该应用程序有一个 AI 助手，使用基于云的模型根据我们的个性化需求推荐最佳航班。

接下来，我们将打开一个酒店预订 web 应用，根据到达日期寻找住宿。该应用程序也有一个 AI 助手，但它使用本地设备模型根据个人信息确定我们理想的酒店要求。完成后，它将这些信息发送给基于云的模型，以为我们推荐最佳的酒店选项。

### 4.2 预订航班和酒店的架构

![Booking App](WebHybridAI.png 'Booking Flight & Hotel Architecture')

如上图所示，航班预订应用程序和酒店预订应用程序的架构如下。使用 Web API 进行混合 AI 的应用流程可分为以下步骤：

1. 航班预订应用中的 Flight Assistant 收集用户需求，并向远程 cloud-based 模型发送请求，获取符合用户偏好的航班信息。

2. 获取航班信息后，Flight Assistant 使用 存储 API 将航班数据转化为向量化的 Categorized Entries。这些向量存储在 Chroma 数据库中。

3. 酒店预订应用中的 Hotel Assistant 收集用户信息，通过 Storage API 将其向量化，并作为 Categorized Entries 同样存储在 Chroma 数据库中。

4. Hotel Assistant 然后向本地 on-device 模型发送请求，结合用户信息和航班信息，确定用户的酒店偏好。

5. Hotel Assistant 再将酒店偏好发送至 cloud-based 模型，该模型返回符合用户需求的酒店列表。

这一过程通过结合 on-device 和 cloud-based 模型，确保了航班和酒店建议的高度个性化。

### 4.3 预订航班和酒店演示

以下演示模拟了两个独立的应用程序与本地和远程 AI 模型的交互，推荐航班和酒店。

![Booking Demo](demo.gif 'Booking Flight & Hotel Demo')

演示分为以下步骤：

1. 左侧是航班预订应用，访问地址为 http://127.0.0.1:5500。当点击 `Send Request` 按钮时，它会向远程 `Gemini` 模型发送请求。由于 LLM 无法获取实时航班数据，此演示仅要求 `Gemini` 预测航班的到达时间。

2. `Gemini` 模型的结果显示为蓝色文本。通过点击 `Insert Entry`，可以将部分蓝色文本（代表航班详情）存储到 `Chroma` 向量数据库中。

3. 点击 `Query Entry` 按钮，显示刚刚保存的航班信息，以确认信息正确。（注意：此按钮使用的 API 不属于 `Storage API`，仅用于演示目的。）

4. 右侧是酒店预订应用，访问地址为 http://localhost:3000，使用不同的端口（5500），模拟两个独立的应用程序。

5. 在酒店预订应用中，点击 `Insert Entry` 模拟将用户的个人信息存储到 `Chroma` 量数据库中。同样，可以点击 `Query Entry` 检查信息是否保存正确。

6. 最后，当点击酒店预订应用的 `Send Request` 按钮时，它会向本地 `Llama` 模型发送请求，要求根据用户的个人信息和航班目的地推荐一系列酒店。

7. `Llama` 模型的结果以蓝色文本显示，列出基于用户预算、目的地等详细信息的酒店推荐，并解释为何选择该酒店。该演示简化了架构中提到的向远程模型发送请求的步骤。

在 GitHub 上该演示的源代码： [Flight Demo](https://github.com/kevinmoch/web-hybrid-ai/blob/main/chromaDB/flight.html) and [Hotel Demo](https://github.com/kevinmoch/web-hybrid-ai/blob/main/chromaDB/hotel.html).

### 4.4 连接 API 演示

在上一节的 Connection API 示例用法 中，我们了解到在以下连接配置中，通常我们首先尝试连接 Gemini 模型。如果失败，我们会依次尝试连接 Llama、GeminiNano 和 Gemma 模型。

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

下方的动画展示了当我们调用 connect 方法时会发生什么。它会依次尝试连接 Gemini、Llama 和 GeminiNano 模型，但都失败了。最终，它成功连接到 Gemma 模型。为了让浏览器控制台中的过程更清晰，我们添加了一些信息提示，显示连接成功或失败的情况。

![Connection Demo](connect.gif 'Connection API Demo')

在 GitHub 上该演示的源代码： [Unified Demo](https://github.com/kevinmoch/web-hybrid-ai/blob/main/webAI/unified.html).

## 5 API 的考虑事项

关于 Web API 混合 AI，有几个方面值得我们深入考虑。

### 5.1 连接策略

除了现有的 Connection API，还应考虑以下潜在的连接策略：

- **连接超时设置：**

  - 为每个模型单独分配连接超时时间，这样如果某个模型响应过慢，可以跳过或重试。
  - 或者，提供一个全局的超时设置，统一应用于所有模型，确保连接多个模型时的行为一致性。

- **自定义连接策略：**

  - 公开一个用户定义的函数，允许开发人员根据各种条件（例如用户偏好、数据大小、模型能力等）创建自定义逻辑来决定访问哪个模型。
  - 这赋予开发人员更细致的控制能力，使得能够根据场景动态选择 `on-device` 或 `cloud-based` 模型。

- **模型状态信息：**
  为开发者提供检查每个已配置模型状态的能力，在发送请求之前包括以下信息：

  - 模型是否可用并能被访问。
  - 模型当前是否忙于处理其他请求。
  - 模型是否处于活动或空闲状态，以便决定是等待模型还是切换到其他模型。

- **跨模型的负载平衡：**
  引入负载平衡机制，自动基于模型的当前负载、可用性和性能分配请求，确保在繁忙系统中的顺畅操作。

- **模型版本控制和兼容性检查：**
  确保系统能够处理不同版本的模型，检查兼容性，并根据需要自动切换到最新或最兼容的版本。

### 5.2 存储策略

除了现有的与本地 Chroma 向量数据库集成的 Storage API 外，还应考虑其他几种存储策略，以确保灵活性、效率和可靠性。以下是一些潜在的存储策略：

- **利用现有的浏览器存储：**
  使用浏览器的原生存储机制，如 localStorage 或 IndexedDB，在用户的本地设备上存储条目信息。这使得无需完整的向量数据库即可轻量存储用户特定数据。

- **可配置的多个本地向量数据库：**
  类似于 Connection API，提供开发者配置多个本地向量数据库的选项。每个 LLM 都可以根据用例或模型要求与特定的向量存储配对。

- **容量管理：**
  在存储数据为公共或共享条目时，需考虑存储限制，尤其是在存储空间受限的环境中（例如基于浏览器的存储或低资源设备）。

- **数据库故障的错误处理：**
  自动重试失败的存储操作或切换到备用存储方法。如果发生数据库损坏，通知用户或开发者，并提供恢复选项，例如自动备份或回滚到之前的稳定状态。

- **支持不同的向量存储格式：**
  通过允许以不同格式存储向量（如 JSON 用于简单结构，或二进制格式用于高效存储和检索大型向量数据集），为开发者提供灵活性。

### 5.3 原生操作系统 API

如果 Connection API 和 Storage API 受到操作系统级别的原生支持（或浏览器只是调用这些底层的原生操作系统 API），它将带来几个好处：

- **本地设备上的跨应用数据共享：**
  无论是在浏览器中运行的 Web 应用程序，还是设备上的原生应用程序（如 iOS、Android 或 Windows 应用程序），这些应用都可以共享它们收集的用户数据，并将其本地存储在设备上。这意味着设备上的任何应用程序——无论是 Web 应用还是原生应用——都可以在使用本地模型进行推理时利用这些共享信息。因此，模型的响应将更加个性化，并与用户的需求保持一致。

- **跨设备无缝体验：**
  假设用户拥有多个运行相同操作系统或来自同一制造商的设备，如 HarmonyOS 或 MacOS & iOS。当用户登录这些设备时，操作系统可以识别用户帐户下的所有设备。通过用户的同意，操作系统可以通过安全的传输通道在设备之间安全共享数据。这将确保当每个设备上的不同本地模型执行推理时，结果能够与用户的个性化偏好保持一致，无论使用的是哪个设备。

- **隐私和安全性提升：**
  由于数据共享发生在操作系统级别，因此它将受益于操作系统的内置安全协议。例如，可以使用端到端加密或安全加密区，确保只有授权的应用程序和模型可以访问数据，在降低隐私风险的同时，仍允许无缝的跨应用功能。

- **统一的数据管理：**
  用户不需要手动在应用程序或平台之间移动他们的数据。操作系统可以集中管理数据，使应用程序能够访问最相关、最新的信息，从而在不同设备和应用程序之间产生更连贯和个性化的模型响应。

- **更好的性能：**
  使用操作系统级别的原生 API 可以加快数据存储和检索过程的速度和效率。通过消除基于浏览器的存储机制的开销，系统可以存储更大体量的数据，并处理更复杂的操作，尤其是在处理向量数据库或大型 AI 模型时。
