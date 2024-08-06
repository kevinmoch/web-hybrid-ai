import express from 'express'
import process from 'child_process'

const app = express()
const port = 3000
const port2 = 5500

app.use(express.static('./'))

app.listen(port, () => {
  process.exec(`start http://localhost:${port}`)
  console.log(`Server listening on port ${port}`)
})

app.listen(port2, () => {
  process.exec(`start http://127.0.0.1:${port2}`)
  console.log(`Server listening on port ${port2}`)
})
