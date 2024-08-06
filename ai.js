import { insertEntry, updateEntry, removeEntry } from './indexedDB/entry.js'
import { connectRemoteModel as gemini, connectLocalModel as gemma } from './model.js'
import llama from './chromaDB/model.js'
import geminiNano from './webAI/model.js'

const models = { gemini, gemma, llama, geminiNano }

const tryConnect = async (prior) => {
  let model = null
  let connect = null

  for (let i = 0; i < prior.length; i++) {
    const [name, connectModel, options] = prior[i]
    try {
      model = await connectModel(options)
    } catch (error) {
      console.error('An error occurs when connecting the model', name, '\n', error)
    }
    if (model) {
      console.warn('Connect model', name, 'successfully')
      connect = connectModel
      break
    }
  }

  return [model, connect]
}

const switchModelFn = (prior, remotes, locals) => async (modelName) => {
  const connectModel = models[modelName]
  const options = remotes[modelName] || locals[modelName]
  const model = await connectModel(options)
  return createSession(model, connectModel, prior, remotes, locals)
}

const promptFn = (model, connect, prior) => {
  return async (...args) => {
    try {
      return await model.generateResponse.apply(model, args)
    } catch (error) {
      console.error('Prompt failed when using the model\n', error)

      for (let i = 0; i < prior.length; i++) {
        const [name, connectModel, options] = prior[i]
        if (connect !== connectModel) {
          try {
            const subModel = await connectModel(options)
            console.warn('Prompt failed, switch the model', name, 'successfully')
            return await subModel.generateResponse.apply(subModel, args)
          } catch (error) {
            console.error('Prompt failed, an error occurs when switching the model', name, '\n', error)
          }
        }
      }
    }
  }
}

const createSession = (model, connect, prior, remotes, locals) => {
  if (model) {
    return {
      prompt: promptFn(model, connect, prior),
      switchModel: switchModelFn(prior, remotes, locals)
    }
  } else {
    throw new Error('No avaliable model can be connected!')
  }
}

const connect = async ({ remotes = {}, remote_priority, locals = {}, local_priority, prefer } = {}) => {
  const remoteNames = remote_priority || Object.keys(remotes)
  const remote = remoteNames.map((name) => [name, models[name], remotes[name]])

  const localNames = local_priority || Object.keys(locals)
  const local = localNames.map((name) => [name, models[name], locals[name]])

  const prior = prefer === 'local' ? local.concat(remote) : remote.concat(local)
  const [model, connect] = await tryConnect(prior)

  return createSession(model, connect, prior, remotes, locals)
}

const ai = { connect, insertEntry, updateEntry, removeEntry }

export default ai
