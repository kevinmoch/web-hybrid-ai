import { createRetrievalChain } from 'langchain/chains/retrieval'
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents'
import { PromptTemplate } from '@langchain/core/prompts'
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama'
import { Ollama } from '@langchain/community/llms/ollama'
import { Chroma } from '@langchain/community/vectorstores/chroma'
import { ChromaClient } from 'chromadb'

const collectionName = 'test'

const pageContent = 'TinyEngine is a lowcode engine that enables developers to customize lowcode platforms.'

const metadata = { category: 'activity' }

const template = 'You are an expert, you are already aware of these: {context}, hence you can answer this question {input}'

const ollamaSetting = { model: 'llama3:8b' }

const question = 'what is TinyEngine'

async function testOllama() {
  const llm = new Ollama(ollamaSetting)
  const stream = await llm.stream(question)

  for await (const chunk of stream) {
    process.stdout.write(chunk)
  }
  console.log('\n' + '='.repeat(100) + '\n')
}

async function setChromaDoc(content) {
  const client = new ChromaClient()
  await client.deleteCollection({ name: collectionName })

  const embeddings = new OllamaEmbeddings(ollamaSetting)
  const vectorStore = new Chroma(embeddings, { collectionName })
  await vectorStore.addDocuments([{ pageContent, metadata }])

  const response = await vectorStore.similaritySearch(content, 4, metadata)
  console.log(response.map((entry) => entry.pageContent).join(','))
  console.log('\n' + '='.repeat(100) + '\n')
}

async function runLangChain(input, filter) {
  const llm = new Ollama(ollamaSetting)
  const embeddings = new OllamaEmbeddings(ollamaSetting)
  const vectorStore = await Chroma.fromExistingCollection(embeddings, { collectionName, filter })

  const prompt = PromptTemplate.fromTemplate(template)
  const combineDocsChain = await createStuffDocumentsChain({ llm, prompt })

  const chain = await createRetrievalChain({
    retriever: vectorStore.asRetriever(),
    combineDocsChain
  })

  const stream = await chain.stream({ input })
  for await (const chunk of stream) {
    if (chunk.answer) {
      process.stdout.write(chunk.answer)
    }
  }
}

await testOllama()
await setChromaDoc(question)
await runLangChain(question, metadata)
