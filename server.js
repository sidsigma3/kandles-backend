require('dotenv').config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();

const httpServer = require("http").createServer(app);
app.use(express.json());
app.use(cors());

const socketIO = require("socket.io");
const io = socketIO(httpServer, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type"],
  },
});



const routeurl = require("./route")(io);
app.use("/", routeurl);
// io.on("connection", (socket) => {
//   console.log("Client connected");

//   socket.on("disconnect", () => {
//     console.log("Client disconnected");
//   });
// });

const PORT = process.env.PORT || 5000;

httpServer.listen(PORT, () => {
  console.log("Server is running on port" , PORT);
});
