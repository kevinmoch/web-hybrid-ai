import { ChromaClient } from 'https://cdn.jsdelivr.net/npm/chromadb/+esm'
import { Chroma } from 'https://cdn.jsdelivr.net/npm/@langchain/community/vectorstores/chroma/+esm'
import { OllamaEmbeddings } from 'https://cdn.jsdelivr.net/npm/@langchain/community/embeddings/ollama/+esm'

const collectionName = 'opentiny'

const ollamaSetting = { model: 'llama3:8b' }

const embeddings = new OllamaEmbeddings(ollamaSetting)

const insertEntry = async (category, content) => {
  const vectorStore = new Chroma(embeddings, { collectionName })
  const ids = await vectorStore.addDocuments([
    {
      pageContent: content,
      metadata: { category }
    }
  ])
  return ids[0]
}

const updateEntry = async (entryId, content) => {
  const vectorStore = new Chroma(embeddings, { collectionName })
  const ids = await vectorStore.addDocuments([{ pageContent: content }], { ids: [entryId] })
  return ids[0] === entryId
}

const removeEntry = async (entryId) => {
  const vectorStore = new Chroma(embeddings, { collectionName })
  await vectorStore.delete({ ids: [entryId] })
  return true
}

const queryEntry = async (category) => {
  const client = new ChromaClient()
  try {
    const collection = await client.getCollection({ name: collectionName })
    const result = await collection.get({ where: { category } })
    return result.documents.join('\n')
  } catch (error) {
    return error.message
  }
}

const clearAll = async () => {
  const client = new ChromaClient()
  try {
    await client.deleteCollection({ name: collectionName })
    return true
  } catch (error) {
    return false
  }
}

export { insertEntry, updateEntry, removeEntry, queryEntry, clearAll }
