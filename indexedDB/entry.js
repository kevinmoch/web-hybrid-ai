const dbName = 'personalizedData'
const storeName = 'entries'
const index = 'category'

const openDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, 1)
    request.onupgradeneeded = (event) => {
      const db = event.target.result
      if (!db.objectStoreNames.contains(storeName)) {
        const store = db.createObjectStore(storeName, { keyPath: 'id' })
        store.createIndex(index, index, { unique: false })
      }
    }
    request.onsuccess = (event) => resolve(event.target.result)
    request.onerror = (event) => reject(event.target.error)
  })
}

const insertEntry = async (category, content) => {
  const id = `${category}-${Date.now()}`
  const entry = { id, category, content }
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite')
    const store = transaction.objectStore(storeName)
    const request = store.add(entry)
    request.onsuccess = () => resolve(id)
    request.onerror = () => reject(null)
  })
}

const updateEntry = async (entryId, content) => {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite')
    const store = transaction.objectStore(storeName)
    const request = store.get(entryId)
    request.onsuccess = (event) => {
      const entry = event.target.result
      if (entry) {
        entry.content = content
        const updateRequest = store.put(entry)
        updateRequest.onsuccess = () => resolve(true)
        updateRequest.onerror = () => reject(false)
      } else {
        reject(false)
      }
    }
    request.onerror = () => reject(false)
  })
}

const removeEntry = async (entryId) => {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite')
    const store = transaction.objectStore(storeName)
    const request = store.delete(entryId)
    request.onsuccess = () => resolve(true)
    request.onerror = () => reject(false)
  })
}

const queryEntry = async (category) => {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readonly')
    const store = transaction.objectStore(storeName)
    const data = store.index(index)
    const request = data.getAll(category)
    request.onsuccess = (event) => {
      const result = event.target.result.reverse()
      resolve(result.map((entry) => entry.content).join('\n'))
    }
    request.onerror = () => reject(null)
  })
}

const clearAll = async () => {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite')
    const store = transaction.objectStore(storeName)
    const request = store.clear()
    request.onsuccess = () => resolve(true)
    request.onerror = () => reject(false)
  })
}

export { insertEntry, updateEntry, removeEntry, queryEntry, clearAll }
