const connectGeminiNano = async (options = {}) => {
  const canCreate = window.ai && (await window.ai.canCreateTextSession())

  if (!canCreate || canCreate === 'no') {
    throw 'Current browser does NOT support window.ai'
  }

  const generateResponse = async (input, display) => {
    try {
      const session = await window.ai.createTextSession(options)
      const stream = session.promptStreaming(input)

      let previousLength = 0
      for await (const chunk of stream) {
        display(chunk.slice(previousLength))
        previousLength = chunk.length
      }
      display('', true)
    } catch (error) {
      throw error.message
    }
  }

  return { generateResponse }
}

export default connectGeminiNano
