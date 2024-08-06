import { createRetrievalChain } from 'https://cdn.jsdelivr.net/npm/langchain/chains/retrieval/+esm'
import { createStuffDocumentsChain } from 'https://cdn.jsdelivr.net/npm/langchain/chains/combine_documents/+esm'
import { PromptTemplate } from 'https://cdn.jsdelivr.net/npm/@langchain/core/prompts/+esm'
import { OllamaEmbeddings } from 'https://cdn.jsdelivr.net/npm/@langchain/community/embeddings/ollama/+esm'
import { Ollama } from 'https://cdn.jsdelivr.net/npm/@langchain/community/llms/ollama/+esm'
import { Chroma } from 'https://cdn.jsdelivr.net/npm/@langchain/community/vectorstores/chroma/+esm'

const collectionName = 'opentiny'

const template = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say you don't know. DO NOT try to make up an answer. If the question is not related to the context,
politely respond that you are tuned to only answer questions that are related to the context. {context}  Question: {input} Helpful answer:`

const ollamaSetting = { model: 'llama3:8b' }

const connectLlama = async (options = {}) => {
  const llm = new Ollama({ ...ollamaSetting, ...options })
  const embeddings = new OllamaEmbeddings(ollamaSetting)

  const prompt = PromptTemplate.fromTemplate(template)
  const combineDocsChain = await createStuffDocumentsChain({ llm, prompt })

  const generateResponse = async (input, display, category) => {
    try {
      const vectorStore = await Chroma.fromExistingCollection(embeddings, { collectionName, filter: { category } })
      const chain = await createRetrievalChain({
        retriever: vectorStore.asRetriever(),
        combineDocsChain
      })

      const stream = await chain.stream({ input })
      for await (const chunk of stream) {
        if (chunk.answer) {
          display(chunk.answer)
        }
      }
      display('', true)
    } catch (error) {
      throw error.message
    }
  }

  return { generateResponse }
}

export default connectLlama
