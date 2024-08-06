import { connectRemoteModel, connectLocalModel, displayResult, clearResult } from './model.js'

const category = 'travel'

const content = 'My personal information: As a female college student traveling with a daily budget of $200.'

const flightRequest = `Cathay Pacific flight CX840 departs from Hong Kong at 4:20 p.m. on August 1, 2024, 
and is expected to fly for 16 hours. tell me the estimated local time in New York JFK airport when it arrives?`

const hotelRequest = `Based on my personal information and travel destination, 
suggest one hotel in the city which is suitable for me.`

const insert = (insertEntry, ctx) => async (e) => {
  e.target.disabled = true
  ctx.vue.entryId = await insertEntry(category, ctx.vue.content)
  e.target.disabled = false
  notify(`Insert entry ${ctx.vue.entryId ? 'successfully' : 'failed'}!`)
}

const update = (updateEntry, ctx) => async (e) => {
  e.target.disabled = true
  const result = await updateEntry(ctx.vue.entryId, ctx.vue.content)
  e.target.disabled = false
  notify(`Update entry ${result ? 'successfully' : 'failed'}!`)
}

const remove = (removeEntry, ctx) => async (e) => {
  if (confirm('Do you want to remove the entry?')) {
    e.target.disabled = true
    const result = await removeEntry(ctx.vue.entryId)
    e.target.disabled = false
    notify(`Remove entry ${result ? 'successfully' : 'failed'}!`)
  }
}

const query = (queryEntry) => async (e) => {
  e.target.disabled = true
  const result = await queryEntry(category)
  e.target.disabled = false
  notify(result)
}

const clear = (clearAll) => async (e) => {
  if (confirm('Do you want to clear all entries?')) {
    e.target.disabled = true
    const result = await clearAll()
    e.target.disabled = false
    notify(`Clear all ${result ? 'successfully' : 'failed'}!`)
  }
}

const flight = ({ insertEntry, updateEntry, removeEntry, queryEntry, clearAll }) => {
  return ({ dom = '#app', connectModel, unifiedApi } = {}) => {
    let ctx = {}

    const app = Vue.createApp({
      data() {
        ctx.vue = this
        return {
          content: '',
          flightRequest,
          remoteModel: null,
          entryId: ''
        }
      },
      methods: {
        async sendRequest(e) {
          const button = e.target
          const output = document.getElementById('flightRemoteOutput')
          clearResult(output, button, 'Sending request...')

          const request = this.flightRequest
          const display = displayResult(output, button, (result) => {
            const lines = result.split('\n').reverse()
            const dest = lines.find((line) => line.includes('2024'))
            if (dest) {
              this.content = 'My travel destination: ' + dest
              for (let btn of document.querySelectorAll('.entry')) btn.disabled = false
            } else {
              notify('Unable to retrieve valid flight information!\nYou need to resend the request.')
            }
          })

          if (unifiedApi) {
            await unifiedApi(request, display)
          } else {
            try {
              const connect = connectModel || connectRemoteModel
              this.remoteModel = this.remoteModel || (await connect())
              await this.remoteModel.generateResponse(request, display)
            } catch (error) {
              display(error, console.error)
            }
          }
        },
        insert: insert(insertEntry, ctx),
        update: update(updateEntry, ctx),
        remove: remove(removeEntry, ctx),
        query: query(queryEntry),
        clear: clear(clearAll)
      }
    })

    app.mount(dom)
  }
}

const hotel = ({ insertEntry, updateEntry, removeEntry, queryEntry, clearAll }) => {
  return ({ dom = '#app', connectModel, needCategory, unifiedApi } = {}) => {
    let ctx = {}

    const app = Vue.createApp({
      data() {
        ctx.vue = this
        return {
          content,
          hotelRequest,
          localModel: null,
          entryId: ''
        }
      },
      methods: {
        async sendRequest(e) {
          const button = e.target
          const output = document.getElementById('hotelLocalOutput')
          clearResult(output, button, 'Sending request...')

          const request = needCategory ? (await queryEntry(category)) + '\n' + this.hotelRequest : this.hotelRequest
          const display = displayResult(output, button)

          if (unifiedApi) {
            await unifiedApi(request, display)
          } else {
            try {
              const connect = connectModel || connectLocalModel
              this.localModel = this.localModel || (await connect())

              if (needCategory) {
                await this.localModel.generateResponse(request, display)
              } else {
                await this.localModel.generateResponse(request, display, category)
              }
            } catch (error) {
              display(error, console.error)
            }
          }
        },
        insert: insert(insertEntry, ctx),
        update: update(updateEntry, ctx),
        remove: remove(removeEntry, ctx),
        query: query(queryEntry),
        clear: clear(clearAll)
      }
    })

    app.mount(dom)
  }
}

const load = async (url) => {
  return new Promise((resolve, reject) => {
    fetch(url)
      .then((response) => response.text())
      .then(resolve)
      .catch(reject)
  })
}

const notify = (message) => {
  if (Notification.permission === 'granted') {
    return new Notification(message)
  } else if (Notification.permission !== 'denied') {
    Notification.requestPermission().then((permission) => {
      if (permission === 'granted') {
        return new Notification(message)
      }
    })
  }
  alert(message)
}

export { load, flight, hotel }
