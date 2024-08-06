import { GoogleGenerativeAI } from 'https://cdn.jsdelivr.net/npm/@google/generative-ai/+esm'
import { FilesetResolver, LlmInference } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai'
import gemini_api_key from './key.js'

const connectRemoteModel = async (options = {}) => {
  const genAI = new GoogleGenerativeAI(gemini_api_key)

  const defaultOption = { model: 'gemini-1.5-flash' }

  const model = genAI.getGenerativeModel({
    ...defaultOption,
    ...options
  })

  const generateResponse = async (content, display) => {
    try {
      const result = await model.generateContentStream([content])
      for await (const chunk of result.stream) {
        display(chunk.text())
      }
      display('', true)
    } catch (error) {
      throw error.message
    }
  }

  return { generateResponse }
}

const connectLocalModel = async (options = {}) => {
  const genAI = await FilesetResolver.forGenAiTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm')

  const defaultOption = {
    baseOptions: { modelAssetPath: '/models/gemma-1.1-2b-it-gpu-int4.bin' },
    maxTokens: 5120,
    randomSeed: 1,
    topK: 1,
    temperature: 1.0
  }

  try {
    const model = await LlmInference.createFromOptions(genAI, {
      ...defaultOption,
      ...options
    })

    const generateResponse = (content, display) => model.generateResponse(content, display)

    return { generateResponse }
  } catch (error) {
    throw error.message
  }
}

const autoResizeOutput = (output) => {
  if (output.textContent) {
    output.style.display = 'block'
    output.style.height = 'auto'
    output.style.height = output.scrollHeight + 'px'
  } else {
    output.style.display = 'none'
  }
}

const displayResult = (output, button, callback) => (partialResults, complete) => {
  output.style.color = 'blue'
  output.textContent += partialResults.replaceAll('*', '')

  if (complete) {
    button.disabled = false
    button.innerText = button.title
    if (typeof complete === 'function') {
      complete(output.textContent)
      output.style.color = 'red'
    } else if (typeof callback === 'function') {
      callback(output.textContent)
    }
  }

  autoResizeOutput(output)
}

const clearResult = (output, button, message) => {
  button.disabled = true
  button.innerText = message
  output.textContent = ''
  autoResizeOutput(output)
}

export { connectRemoteModel, connectLocalModel, displayResult, clearResult }
