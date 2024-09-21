const express = require("express");
const router = express.Router();
const mysql = require("mysql");
const db = require("./config/db");
const nodemailer = require("nodemailer");
var url = require("url");
const emailId = "sidsigma3@gmail.com";
const pass = "zsswervkokmbocgl";
const fs = require("fs");
const handlebars = require("handlebars");
const hbs = require("nodemailer-express-handlebars");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");
const { error } = require("console");
const axios = require("axios");
const greeks = require("greeks");
const readline = require("readline");
const open = require("open");
const axiosCookieJarSupport = require("axios-cookiejar-support").default;
const tough = require("tough-cookie");
const KiteConnect = require("kiteconnect").KiteConnect;
const KiteTicker = require("kiteconnect").KiteTicker;
// const WebSocket = require("ws");
const io = require("./server");
const { v4: uuidv4 } = require('uuid');
// const Upstox = require('upstox-js-sdk');
const puppeteer = require('puppeteer');
const otplib = require('otplib');
const Razorpay = require('razorpay');
const path = require('path');
const { Builder, By, until } = require('selenium-webdriver');
const chrome = require('selenium-webdriver/chrome');
const { spawn } = require('child_process');



const welcomeEmailPath = path.join(__dirname, "config/WelocomeEmail.html");
const resetEmailPath = path.join(__dirname, "config/resetEmail.html");



// Now, read the files using the absolute paths
const htmlTemplate = fs.readFileSync(welcomeEmailPath, "utf8");
const resetEmail = fs.readFileSync(resetEmailPath, "utf-8");

var access_token = null;
let protobufRoot = null;
const WebSocket = require("ws");
const protobuf = require("protobufjs");

// const { createReadStream } = require('fs');
const csv = require('csv-parser');

const cookieJar = new tough.CookieJar();



// const stream = fs.createReadStream('config/instrument.csv', 'utf8');
const razorpay = new Razorpay({
  key_id: 'rzp_test_u9YDoBJGjerFbp',
  key_secret: 'tD1op5y9kL0Xa9S6gnbeXrMY',
});


const sendEmail = async (recipient, userName) => {
  // create reusable transporter object using the default SMTP transport

  const source = htmlTemplate;
  const template = handlebars.compile(source);
  const replacements = {
    username: userName,
    login_url: recipient,
  };
  const htmlToSend = template(replacements);

  let transporter = nodemailer.createTransport({
    host: "smtp.gmail.com",
    port: 465,
    secure: true, // use SSL
    auth: {
      user: emailId,
      pass: pass,
    },
  });

  // send mail with defined transport object
  let info = await transporter.sendMail({
    from: emailId,
    to: recipient,
    subject: "Welcome Message",
    text: "welcome , How you doing?",
    html: htmlToSend,
  });

  console.log("Message sent: %s", info.messageId);
};
let loginAttempts = 0;

module.exports = function (io) {


    router.post("/login", (req, res) => {
    const email = req.body.email;
    const password = req.body.password;

    // check if the user has attempted to login 3 times
    if (loginAttempts >= 3) {
      res.json({
        stat: 401,
        msg: "You have exceeded the maximum number of login attempts. Please wait for 3 minute or try forgot password.",
      });
      setTimeout(() => {
        loginAttempts = 0;
      }, 60000);
      return;
    }

    db.query(
      "SELECT * FROM bjbjotkpn4piwqplzpwn.user WHERE email=? AND password=?",
      [email, password],
      (err, result) => {
        if (err) {
          console.log(err);
          res.json({
            stat: 500,
            msg: "Server error",
          });
        } else {
          if (result.length > 0) {

            const token = jwt.sign({ email: result[0].email, userId: result[0].id }, 'secret-key', { expiresIn: '1h' });

            loginAttempts = 0;
            res.json({
              stat: 200,
              msg: "Sucessfully entered website",
              token: token,
            });
          } else {
            loginAttempts++;
            res.json({
              stat: 201,
              msg:
                "Email and password doesn't match you have, " +
                (4 - loginAttempts) +
                " more attempts left",
            });
          }
        }
      }
    );
  });


  router.post('/payment', async (req,res)=>{

    const amount = Number(req.body.price +'00')


    const options = {
      amount: amount,
      currency: 'INR',
      receipt: 'receipt_order_74394',
      payment_capture: 1,
    };


    try {
      const response = await razorpay.orders.create(options);
      res.json(response);
    } catch (error) {
      console.log(error);
    }
  })


  router.post('/api/instruments',(req,res)=>{
    const instrumentNames = [];
    const stream = fs.createReadStream('./config/instrument.csv', 'utf8');
    // csv.parse(instrumentfile, { headers: true })
    // .on('data', (row) => {
    //   // Assuming your instrument names are in the 'name' column
    //   if (row.name) {
    //     instrumentNames.push(row.name);
    //   }
    // })
    // .on('end', () => {
    //   res.json({ instrumentNames });
    // });

    stream.pipe(csv())
    .on('data', (row) => {
      // Assuming your instrument names are in the 'name' column
      if (row.name && row.instrument_key) {
        const instrument = {
          name: row.name,
          instrument_key: row.instrument_key
        };
        instrumentNames.push(instrument);
      }
    })
    .on('end', () => {
      res.json({ instrumentNames });
    })
    .on('error', (error) => {
      console.error('Error processing CSV:', error);
      res.status(500).json({ error: 'Internal Server Error' });
    })

  })

  router.post('/api/instrumentslist',(req,res)=>{
    const instrumentNames = [];
    const stream = fs.createReadStream('./config/instruments.csv', 'utf8');
    // csv.parse(instrumentfile, { headers: true })
    // .on('data', (row) => {
    //   // Assuming your instrument names are in the 'name' column
    //   if (row.name) {
    //     instrumentNames.push(row.name);
    //   }
    // })
    // .on('end', () => {
    //   res.json({ instrumentNames });
    // });

    stream.pipe(csv())
    .on('data', (row) => {
      // Assuming your instrument names are in the 'name' column
      if (row.name && row.tradingsymbol) {
        const instrument = {
          name: row.name,
          instrument_key: row.tradingsymbol,
          expiry:row.expiry
        };
        instrumentNames.push(instrument);
      }
    })
    .on('end', () => {
     
      res.json({ instrumentNames });
    });

  })



  router.post("/get-master-quote", async (req, res) => {
    // Initialize a cookie jar
    const cookieJar = new tough.CookieJar();
    
    // Define headers with the cookie jar
    const headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept-Encoding': 'gzip, deflate, br',
      'Referer': 'https://www.nseindia.com/',
      'X-Requested-With': 'XMLHttpRequest',
      'Connection': 'keep-alive',
    };
  
    // Make the initial request to obtain cookies
    try {
      const response = await axios.get('https://www.nseindia.com', { headers });
      const cookies = response.headers['set-cookie'];
  
      // Set cookies in the cookie jar
      cookies.forEach(cookie => {
        cookieJar.setCookieSync(cookie, 'https://www.nseindia.com');
      });
    } catch (error) {
      console.error('Error setting initial cookies:', error);
      return res.status(500).send('Server Error');
    }
  
    // Use the cookie jar for the subsequent request
    const url = 'https://www.nseindia.com/api/master-quote';
  
    try {
      const response = await axios.get(url, {
        withCredentials: true,
        jar: cookieJar,
        headers: {
          ...headers,
          cookie: cookieJar.getCookieStringSync('https://www.nseindia.com'),
        },
      });
  
      // Process the response and send the data back
      const { data } = response;
      console.log(data);
  
      // Send the data back to the client
      res.json(data);
    } catch (error) {
      console.error('Error fetching master quote:', error);
      res.status(500).send('Server Error');
    }
  });


  router.post("/signup", (req, res) => {
    console.log(req.body);
    const registrationMethod = req.body.registrationMethod;
    let firstName =''
    let lastName ='';
    let userName ='' ;
    let email =''
    let password ='';
    let repassword ='';
    let phoneNumber ='';
    let id =''

    let fullName = '';

    if (registrationMethod === "email") {
      // Handle email signup logic
       firstName = req.body.name;
     
       userName = req.body.username;
       email = req.body.email;
       password = req.body.password;
       repassword = req.body.rePassword;
       phoneNumber = req.body.phone;
  
       fullName = firstName 
  
      // Your existing email signup logic here
    } else if (registrationMethod === "google") {
      // Handle Google signup logic
      
       firstName = req.body.name
       userName = req.body.name;
       email = req.body.email;
       phoneNumber = req.body.password;
       password = req.body.password;
     
  
       fullName = firstName 
  
      // Your existing Google signup logic here
    }


  

    db.query(
      "SELECT * FROM login.user WHERE email=? ",
      [email],
      (err, result) => {
        if (err) {
          console.log(err);
        } else {
          if (result.length > 0) {
            res.json({
              stat: 201,
              msg: "user already registered",
            });
          } else {
            console.log(phoneNumber);
            db.query(
              "INSERT INTO login.user (fullName, userName, email,password,phoneNumber) VALUES (?,?,?,?,?)",
              [fullName, userName, email, password, phoneNumber],
              (err, resu) => {
                if (err) {
                  console.log(err);
                } else {
                  res.json({
                    stat: 200,
                    msg: "Succesfully Registered",
                  });

                  sendEmail(email, userName);
                }
              }
            );
          }
        }
      }
    );
  });

  router.post("/reset", (req, res) => {
    const email = req.body.email;

    const token = jwt.sign({ email }, "your-secret-key", { expiresIn: "1h" });

    db.query(
      "INSERT INTO password_reset_tokens (email, token) VALUES (?, ?)",
      [email, token],
      (error, results, fields) => {
        if (error) {
          console.error(error);
          res.status(500).send("Internal server error");
          return;
        }

        const resetLink = `http://localhost:3000/pass?email=${email}&token=${token}`;

        let transporter = nodemailer.createTransport({
          host: "smtp.gmail.com",
          port: 465,
          secure: true, // use SSL
          auth: {
            user: emailId,
            pass: pass,
          },
        });

    
        let mailOptions = transporter
          .sendMail({
            from: "sidsigma3@gmail.com",
            to: email,
            subject: "Password reset request",
            html: `<p>You have requested a password reset for your account. Please click the following link to reset your password:</p><p><a href="${resetLink}">${resetLink}</a></p>`,
          })
          .then(
            res.json({
              stat: 200,
              msg: "Succesfully Registered",
            })
          );
      }
    );
  });

  router.post("/change", (req, res) => {
    const email = req.body.Email;
    const password = req.body.pass;
    const repass = req.body.repass;

    if (password !== repass) {
      res.status(402);
    }

    db.query(
      "UPDATE login.user SET password = ? WHERE email = ?",
      [password, email],
      (error, results, fields) => {
        if (error) {
          console.error(error);
          res.status(500).send("Internal server error");
        } else {
          db.query(
            "DELETE FROM password_reset_tokens WHERE email = ?",
            [email],
            (error, results, fields) => {
              if (error) {
                console.error(error);
              }

              // res.status(200).send('Password changed successfully.');

              const source = resetEmail;
              const template = handlebars.compile(source);
              const replacements = {
                // username: "",
                // login_url:recipient
              };
              const emailToSend = template(replacements);

              let transporter = nodemailer.createTransport({
                host: "smtp.gmail.com",
                port: 465,
                secure: true, // use SSL
                auth: {
                  user: emailId,
                  pass: pass,
                },
              });

              let info = transporter
                .sendMail({
                  from: emailId,
                  to: email,
                  subject: "Your password has been changed",

                  html: emailToSend,
                })
                .then(res.status(200).send("successfull"));
            }
          );
        }
      }
    );
  });

  router.post("/check", (req, res) => {
    const { email, token } = req.body;

    db.query(
      "SELECT * FROM password_reset_tokens WHERE email = ? AND token = ?",
      [email, token],
      (error, results, fields) => {
        if (error) {
          console.error(error);
          res.status(500).send("Internal server error");
          return;
        }

        if (results.length === 0) {
          res.status(400).send("Invalid or expired token");
          return;
        } else {
          res.status(200).json({ msg: true });
        }
      }
    );
  });

  const secret = "1gr9bsa88g1td3o52do2bilh74kw8a73";
  const apiKey = "0bpuke0rhsjgq3lm";
  const kite = new KiteConnect({
    api_key: apiKey,
  });

  // const ticker = new KiteTicker({
  //   api_key: "your_api_key",
  //   access_token: "your_access_token",
  // });

  // ticker.connect();

//   const apiVersion = '2.0'
//  router.post("/getLastPrice",(req,res)=>{

//   const indices = req.body.indices

//   const symbol ='NSE_INDEX|'+indices

//   const parts = symbol.split('|');

// // Combine the parts with ':'
//   const convertedSymbol = parts.join(':');

//  console.log(symbol)

  // let api = new Upstox.MarketQuoteApi();
  // api.getFullMarketQuote(symbol, apiVersion, (error, data, response) => {
  //   if (error) {
  //     console.error('wrong');
  //   } else {
  //     const price= data.data[convertedSymbol].lastPrice
  //     const change = data.data[convertedSymbol].netChange
  //     console.log('API called successfully. Returned data: ' + change);
  //     res.send({lastPrice:price,change:change})
  //   }
  // });





  // getQuote(["NSE"+":" + indices]);
  // let lastPrice = null

  // function getQuote(instruments) {
  //   kite
  //     .getQuote(instruments)
  //     .then(function (response) {
  //       // console.log(response);
  //       const price = response[instruments[0]].last_price;
  //       // instrument.push(token)
  //       lastPrice = price
  //       res.send({'lastPrice':lastPrice})
  //       console.log(response)

  //     })
  //     .catch(function (err) {
  //       console.log(err);
  //     });
  // }  


  // const getMarketFeedUrl = async () => {
  //   return new Promise((resolve, reject) => {
  //     let apiInstance = new Upstox.WebsocketApi(); // Create new Websocket API instance
  
  //     // Call the getMarketDataFeedAuthorize function from the API
  //     apiInstance.getMarketDataFeedAuthorize(
  //       apiVersion,
  //       (error, data, response) => {
  //         if (error) reject(error); // If there's an error, reject the promise
  //         else resolve(data.data.authorizedRedirectUri); // Else, resolve the promise with the authorized URL
  //       }
  //     );
  //   });
  // };
  
  // Function to establish WebSocket connection
  // const connectWebSocket = async (wsUrl) => {
  //   return new Promise((resolve, reject) => {
  //     const ws = new WebSocket(wsUrl, {
  //       headers: {
  //         "Api-Version": apiVersion,
  //         Authorization: "Bearer " + accessToken,
  //       },
  //       followRedirects: true,
  //     });
  
  //     // WebSocket event handlers
  //     ws.on("open", () => {
  //       console.log("connected");
  //       resolve(ws); // Resolve the promise once connected
  
  //       // Set a timeout to send a subscription message after 1 second
  //       setTimeout(() => {
  //         const data = {
  //           guid: "someguid",
  //           method: "sub",
  //           data: {
  //             mode: "full",
  //             instrumentKeys: [symbol],
  //           },
  //         };
  //         ws.send(Buffer.from(JSON.stringify(data)));
  //       }, 1000);
  //     });
  
  //     ws.on("close", () => {
  //       console.log("disconnected");
  //     });
  
      // ws.on("message", (data) => {
      //   const decodedData = decodeProfobuf(data);
      //   console.log(JSON.stringify(decodedData)); // Decode the protobuf message on receiving it
      
      //   // Extract last price and change from decoded data
      //   const lastPrice = decodedData.feeds['NSE_INDEX|Nifty Bank'].ff.indexFF.ltpc.ltp;
      //   const change = decodedData.feeds['NSE_INDEX|Nifty Bank'].ff.indexFF.ltpc.cp;
        
      //   // Emit last price and change to connected clients
      //   io.emit('marketDataUpdate', { lastPrice, change });
      // });
  
  //     ws.on("message", (data) => {
  //       const decodedData = decodeProfobuf(data);
       
      
  //       // Check if the expected properties exist before accessing them
  //       if (
  //         decodedData.feeds &&
  //         decodedData.feeds[symbol] &&
  //         decodedData.feeds[symbol].ff &&
  //         decodedData.feeds[symbol].ff.indexFF &&
  //         decodedData.feeds[symbol].ff.indexFF.ltpc
  //       ) {
  //         // Extract last price and change from decoded data
  //         const lastPrice = decodedData.feeds[symbol].ff.indexFF.ltpc.ltp;
  //         const cp = decodedData.feeds[symbol].ff.indexFF.ltpc.cp;

  //         const priceChange = lastPrice - cp;
  //         const percentageChange = ((lastPrice - cp) / cp) * 100;
          
  //         // Emit last price and change to connected clients
  //         io.emit('marketDataUpdate', { lastPrice, indices ,percentageChange,priceChange});
  //       } else {
  //         console.error('Expected properties not found in the data structure.');
  //       }
  //     });

  //     ws.on("error", (error) => {
  //       console.log("error:", error);
  //       reject(error); // Reject the promise on error
  //     });
  //   });
  // };
  
  // // Function to initialize the protobuf part
  // const initProtobuf = async () => {
  //   protobufRoot = await protobuf.load('./config/MarketDataFeed.proto');
  //   console.log("Protobuf part initialization complete");
    
  // };
  
  // // Function to decode protobuf message
  // const decodeProfobuf = (buffer) => {
  //   if (!protobufRoot) {
  //     console.warn("Protobuf part not initialized yet!");
  //     return null;
  //   }
  
  //   const FeedResponse = protobufRoot.lookupType(
  //     "com.upstox.marketdatafeeder.rpc.proto.FeedResponse"
  //   );
  //   return FeedResponse.decode(buffer);
  // };
  
  // // Initialize the protobuf part and establish the WebSocket connection
  // (async () => {
  //   try {
  //     await initProtobuf(); // Initialize protobuf
  //     const wsUrl = await getMarketFeedUrl(); // Get the market feed URL
  //     const ws = await connectWebSocket(wsUrl); // Connect to the WebSocket
  //   } catch (error) {
  //     console.error("An error occurred:", error);
  //   }
  // })();

//  })  



//  router.post("/getGraph",(req,res)=>{

//   const today = new Date().toISOString().split('T')[0];
//   const selectedInstrument = req.body.instrument || "NSE_INDEX|Nifty 50";
//   let api = new Upstox.HistoryApi();
//   let instrumentKey = selectedInstrument; // String | 
//   let interval = "day"; // String | 
//   let toDate = today; // String | 
//   api.getHistoricalCandleData(instrumentKey, interval, toDate, apiVersion, (error, data, response) => {
//     if (error) {
//       console.error(error);
//     } else {
//       console.log('API called successfully. Returned data: ' + data);
//       res.send({candle:data.data.candles})
//     }
//   });

//  })


 router.post("/top-movers", async (req, res) => {

  const url = 'https://iislliveblob.niftyindices.com/jsonfiles/equitystockwatch/EquityStockWatchNIFTY%2050.json?{}&_=' + Date.now();

axios.get(url, {
  headers: {
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Sec-Fetch-Mode': 'cors',
    // Add other headers as needed
  },
})
  .then(response => {
    console.log(response.data);
    res.send(response.data)
  })
  .catch(error => {
    console.error(error);
    // Handle errors here
  });



 })


router.post("/kiteInstrumentList",async(req,res)=>{
    try{
      const instrumentResponse = await kite.getInstruments();
      const instruments = instrumentResponse;
      res.send({instruments});
    }
    catch (error) {
      console.error('Error fetching data:', error);
      res.status(500).send({ error: 'Something went wrong' });
    }

   

})





 router.post("/user-info", async (req, res) => {
  try {
    // Fetch margin data
    const marginResponse = await getMargins(["equity"]);
    const capital = marginResponse.available.live_balance;

    // Fetch instrument data
    // const instrumentResponse = await kite.getInstruments();
    // const instruments = instrumentResponse;
      
    // Send the response to frontend
    res.send({ capital});
  } catch (error) {
    console.error('Error fetching data:', error);
    res.status(500).send({ error: 'Something went wrong' });
  }
});

// Function to fetch margin data
function getMargins(segment) {
  return kite.getMargins(segment);
}


    let socketConnected = false

    let currentSocket = null;

    router.post("/trade-info", (req, res) => {
     
      const symbol = req.body.index;
      const exchange = req.body.exchange
      let strike = null;
      const info = {};
      const instrument=[]
      
      
        const ticker = new KiteTicker({
          api_key: "0bpuke0rhsjgq3lm",
          access_token: access_token,
        });

          getQuote([exchange+":" + symbol]);

          console.log(symbol,exchange)

          function getQuote(instruments) {
            kite
              .getQuote(instruments)
              .then(function (response) {
                console.log(response);
                const token = response[instruments[0]].instrument_token;
                instrument.push(token)
  
                console.log(instrument)
                
                ticker.connect();
                ticker.on("ticks", onTicks);
                ticker.on("connect", subscribe);
    
              })
              .catch(function (err) {
                console.log(err);
              });
          }  
    
          
        function sock (){
         

          setInterval(()=>{
            strikePrice(strike)}, 2000);
    

          if (socketConnected) {
            // Disconnect the previous socket connection
            io.close();
            socketConnected = false;
          }


          io.on("connection", (socket) => {
                 
            console.log("Client connected");
      
            setInterval(()=>{
              strikePrice(strike)}, 2000);
      
            socket.on("disconnect", () => {
              console.log("Client disconnected");
            });
          });
      
          // socketConnected = true

        }
    
        // sock()
        res.send()
        
    
        function onTicks(ticks) {
         
          currPrice = ticks[0].last_price;
          
          
          ticks.forEach((tick) => {
            const instrumentToken = tick.instrument_token;
            const lastPrice = tick.last_price;
            const open = tick.ohlc.open
            // Update instrument data with the latest last price
            strike =lastPrice
            io.emit('strike',{strike:strike,index:symbol})
            
          })
    
        }
    
        function subscribe() {
          
          ticker.subscribe(instrument);
          ticker.setMode(ticker.modeFull, instrument);
        }
    

        function strikePrice(strike){
          console.log(strike,symbol)
            io.emit('strike',{strike:strike,index:symbol})

        }


        // function getQuote(instruments) {
        //   kite
        //     .getQuote(instruments)
        //     .then(function (response) {
        //       console.log(response);
        //       strike = response[instruments[0]].last_price;
        //       res.send({ strike });
        //     })
        //     .catch(function (err) {
        //       console.log(err);
        //     });
        // }

        function getLTP(instruments) {
          kite
            .getLTP(instruments)
            .then(function (response) {
              console.log(response);
            })
            .catch(function (err) {
              console.log(err);
            });
        }
           
        getLTP(["NSE:" + symbol]);
      });

  const API_KEY = 'fd745e7f-f970-4bc8-b11a-703bb47420dd';
  const API_SECRET = '71z9mo074x';
  const REDIRECT_URI = 'http://localhost:3000/dashboard'; // Make sure this matches the redirect URI in your Upstox app settings
  // const upstox = new Upstox(API_KEY, API_SECRET);
  // const loginUrl = upstox.getLoginUri(REDIRECT_URI);
  // const defaultClient = Upstox.ApiClient.instance;
  // const OAUTH2 = defaultClient.authentications['OAUTH2'];
  // const api = new Upstox.MarketQuoteApi()
  var accessToken
  // const api = new UpstoxClient.ChargeApi()
  // const AUTH_URL = 'https://api-v2.upstox.com/login/authorization/dialog'
  // const accessTokenUrl = 'https://api-v2.upstox.com/login/authorization/token';
  // const authUrl = `${AUTH_URL}?client_id=${API_KEY}&redirect_uri=${REDIRECT_URI}&response_type=code`
 
  const otplibAuthenticator = otplib.authenticator;

  // let apiInstance = new UpstoxClient.HistoryApi();
  // let instrumentKey = "NSE_INDEX|Nifty 50"; // String | 
  const interval = "1d"; // String | 
  let toDate = "2023-11-15"; // String | 
  // let apiVersion = "apiVersion_example"; // String | API Version Header
 
  router.post("/connect/kite", (req, res) => {
    const requestToken = req.body.requestToken;

    const connectKite = async () => {
      try {
        const response = await kite.generateSession(requestToken, secret);
        access_token = response.access_token;
        console.log("processing",access_token);
        await kite.setAccessToken(access_token);
        console.log('hua kya')
        const ticker = new KiteTicker({
          api_key: "0bpuke0rhsjgq3lm",
          access_token: access_token,
        });

        ticker.connect();
        res.send('ho gaya')
      } catch (error) {
        console.error(error);
      }
    };
    if (!access_token) {
      connectKite();
    }

  })

  // router.post("/connect/upstox", (req, res) => {
  //   // const requestToken = req.body.requestToken;

  //   // const connectKite = async () => {
  //   //   try {
  //   //     const response = await kite.generateSession(requestToken, secret);
  //   //     access_token = response.access_token;
  //   //     console.log("processing");
  //   //     await kite.setAccessToken(access_token);

  //   //     const ticker = new KiteTicker({
  //   //       api_key: "0bpuke0rhsjgq3lm",
  //   //       access_token: access_token,
  //   //     });

  //   //     ticker.connect();
  //   //     res.send('ho gaya')
  //   //   } catch (error) {
  //   //     console.error(error);
  //   //   }
  //   // };
  //   // if (!access_token) {
  //   //   connectKite();
  //   // }

    
  //   // const authorizationCode = req.body.requestToken;
  //   // console.log(authorizationCode)
  
  //   // const params = {
  //   //   code: authorizationCode,
  //   //   client_id: API_KEY,
  //   //   client_secret: API_SECRET,
  //   //   redirect_uri: REDIRECT_URI,
  //   //   grant_type: 'authorization_code',
  //   // };
    

  // //   var params = {
  // //     'apiSecret' : API_SECRET,
  // //     'code' : authorizationCode,
  // //     'grant_type' : "authorization_code",
  // //     'redirect_uri' : REDIRECT_URI
  // // };

  //   const headers = {
  //     'accept': 'application/json',
  //     'Api-Version': '2.0',
  //     'Content-Type': 'application/x-www-form-urlencoded',
  //   };


  //   if (authorizationCode) {
  //     // Step 4: Exchange the authorization code for an access token
  //     // upstox.getAccessToken(params)
  //     // .then(function (response) {
  //     //   const accessToken = response.access_token;
  //     //   console.log('Access Token:', accessToken);
  //     // })
  //     // .catch(function (err) {
  //     //   console.error('Error getting access token:', err);
  //     //   // Handle the error as needed
  //     // });

  //     axios.post(accessTokenUrl, new URLSearchParams(params), { headers })
  //   .then(response => {
  //     accessToken = response.data.access_token;
  //     console.log('Access Token:', accessToken);
  //   // Now you can use the access token for making authorized requests to Upstox 
  //     OAUTH2.accessToken = accessToken
  //     res.send({'messeage':'completed'})
  //     // api.getFullMarketQuote(symbol, apiVersion, (error, data, response) => {
  //     //   if (error) {
  //     //     console.error(error);
  //     //   } else {
  //     //     console.log('API called successfully. Returned data: ' + data.data[convertedSymbol].lastPrice);
  //     //   }
  //     // });

  //     })
  //     .catch(error => {
  //       console.error('Error obtaining access token:', error.message);
  //       // Handle the error as needed
  //     });

  //     } else {
  //       console.error('Authorization code not found.');
  //     }
    
  

  // });

  router.post("/stock", async (req, res) => {
 
    const symbol = req.body.symbol;
    const expiryDate = req.body.date;
    const requestToken = req.body.requestToken;
  
    // Initialize a cookie jar
    const cookieJar = new tough.CookieJar();
  
    // Define headers with the cookie jar
    const headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept-Encoding': 'gzip, deflate, br',
      'Referer': 'https://www.nseindia.com/',
      'X-Requested-With': 'XMLHttpRequest',
      'Connection': 'keep-alive',
    };
  
    // Make the initial request to obtain cookies
    try {
      const response = await axios.get('https://www.nseindia.com', { headers });
      const cookies = response.headers['set-cookie'];
  
      // Set cookies in the cookie jar
      cookies.forEach(cookie => {
        cookieJar.setCookieSync(cookie, 'https://www.nseindia.com');
      });
    } catch (error) {
      console.error('Error setting initial cookies:', error);
      return res.status(500).send('Server Error');
    }
  
    // Use the cookie jar for subsequent requests
    const url1 = `https://www.nseindia.com/api/option-chain-equities?symbol=${symbol}&date=${expiryDate}`;
    const url2 = `https://www.nseindia.com/api/option-chain-indices?symbol=${symbol}`;
    const url = (symbol === 'NIFTY' || symbol==='BANKNIFTY' || symbol ==='MIDCPNIFTY' || symbol==='FINNIFTY') ? url2 : url1;

    
    try {
      const response = await axios.get(url, {
        headers: {
          ...headers,
          cookie: cookieJar.getCookieStringSync('https://www.nseindia.com'),
        },
      });
      console.log('stock he kya',response.data)
  
      // Process the response and send the data back
      
      // ... (your existing response processing logic)
  
      const { data } = response;
      const expiries = data.records.expiryDates;
      console.log('data he',data)
      const filteredData = data.filtered.data
      console.log('filtered',filteredData)
      const optionChain = {
        calls: [],
        puts: [],
      };
  
      filteredData.forEach(option => {
        const callOption = {
        strikePrice:option.CE?.strikePrice,
        expiryDate:option.CE?.expiryDate,
        underlying:option.CE?.underlying,
        identifier:option.CE?.identifier,
        openInterest:option.CE?.openInterest,
        changeinOpenInterest:option.CE?.changeinOpenInterest,
        pchangeinOpenInterest:option.CE?.pchangeinOpenInterest,
        totalTradedVolume:option.CE?.totalTradedVolume,
        impliedVolatility:option.CE?.impliedVolatility,
        lastPrice:option.CE?.lastPrice,
        change:option.CE?.change,
        pChange:option.CE?.pChange,
        totalBuyQuantity:option.CE?.totalBuyQuantity,
        totalSellQuantity:option.CE?.totalSellQuantity,
        bidQty:option.CE?.bidQty,
        bidprice:option.CE?.bidprice,
        askQty:option.CE?.askQty,
        askPrice:option.CE?.askPrice,
        underlyingValue:option.CE?.underlyingValue
        };
  
        const putOption = {
          strikePrice:option.PE?.strikePrice,
          expiryDate:option.PE?.expiryDate,
          underlying:option.PE?.underlying,
          identifier:option.PE?.identifier,
          openInterest:option.PE?.openInterest,
          changeinOpenInterest:option.PE?.changeinOpenInterest,
          pchangeinOpenInterest:option.PE?.pchangeinOpenInterest,
          totalTradedVolume:option.PE?.totalTradedVolume,
          impliedVolatility:option.PE?.impliedVolatility,
          lastPrice:option.PE?.lastPrice,
          change:option.PE?.change,
          pChange:option.PE?.pChange,
          totalBuyQuantity:option.PE?.totalBuyQuantity,
          totalSellQuantity:option.PE?.totalSellQuantity,
          bidQty:option.PE?.bidQty,
          bidprice:option.PE?.bidprice,
          askQty:option.PE?.askQty,
          askPrice:option.PE?.askPrice,
          underlyingValue:option.PE?.underlyingValue
          };
          optionChain.calls.push(callOption);
          optionChain.puts.push(putOption);
  
      });
          // console.log(optionChain)
      res.send(optionChain)
   
    } catch (error) {
      console.error('Error making request:', error);
      res.status(500).send('Server Error');
    }
  });
  










  

  router.post("/calculate-greeks", (req, res) => {
    // const apiKey = "OOT5PNL8EV6DJ5J8";
    // const symbol = req.body.symbol;
    // const expiryDate = req.body.date;

    // const optionChain = {
    //   calls: [],
    //   puts: [],
    //   spotPrice: "",
    // };

    // const url = `https://opstra.definedge.com/api/openinterest/optionchain/free/${symbol}&${expiryDate}`;

    // axios
    //   .get(url, {
    //     headers: {
    //       "User-Agent":
    //         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    //       "Accept-Language": "en-US,en;q=0.9",
    //       "Accept-Encoding": "gzip, deflate, br",
    //       // 'Referer': 'https://www.nseindia.com/',
    //       cookie:"_ga=GA1.2.1370922286.1680970843; JSESSIONID=E79D7D0FCA2D2DC9EE3E8199FC58227B",
    //       "X-Requested-With": "XMLHttpRequest",
    //       Connection: "keep-alive",
    //     },
    //   })
    //   .then((response) => {
    //     const { data } = response;

    //     data.data.map((option) => {
    //       const callGreeksData = {
    //         gamma: option.CallGamma,
    //         vega: option.CallVega,
    //         theta: option.CallTheta,
    //         delta: option.CallDelta,
    //         strikePrice: option.StrikePrice,
    //         iv: option.CallIV,
    //       };

    //       optionChain.calls.push(callGreeksData);
    //       const putGreeksData = {
    //         gamma: option.PutGamma,
    //         vega: option.PutVega,
    //         theta: option.PutTheta,
    //         delta: option.PutDelta,
    //         strikePrice: option.StrikePrice,
    //         iv: option.PutIV,
    //       };

    //       optionChain.puts.push(putGreeksData);
    //     });

    //     res.send(optionChain);
    //   })
    //   .catch((error) => console.error(error));

    const apiKey = "OOT5PNL8EV6DJ5J8";
    const symbol = req.body.symbol;
    const expiryDate = req.body.date;
  
    // Initialize a cookie jar
    const cookieJar = new tough.CookieJar();
  
    const optionChain = {
      calls: [],
      puts: [],
      spotPrice: "",
    };
  
    const url = `https://opstra.definedge.com/api/openinterest/optionchain/free/${symbol}&${expiryDate}`;
  
    // Make the initial request to obtain cookies
    axios
      .get(url, {
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
          "Accept-Language": "en-US,en;q=0.9",
          "Accept-Encoding": "gzip, deflate, br",
          // 'Referer': 'https://www.nseindia.com/',
          "X-Requested-With": "XMLHttpRequest",
          Connection: "keep-alive",
        },
      })
      .then((response) => {
        // Extract and set cookies in the cookie jar
        const cookies = response.headers['set-cookie'];
        cookies.forEach(cookie => {
          cookieJar.setCookieSync(cookie, url);
        });
  
        // Make the actual request with the obtained cookies
        axios.get(url, {
          headers: {
            ...headers,
            cookie: cookieJar.getCookieStringSync(url),
          },
        })
        .then((response) => {
          const { data } = response;
  
          data.data.map((option) => {
            // Your existing logic for processing and pushing data to optionChain
          });
  
          res.send(optionChain);
        })
        .catch((error) => console.error(error));
  
      })
      .catch((error) => console.error(error));
  });

  router.post("/oi-changes",async (req, res) => {
   
    const symbol = req.body.symbol;
  const expiryDate = req.body.date;

  // Initialize a cookie jar
  const cookieJar = new tough.CookieJar();

  // Define headers with the cookie jar
  const headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.nseindia.com/',
    'X-Requested-With': 'XMLHttpRequest',
    'Connection': 'keep-alive',
  };

  // Make the initial request to obtain cookies
  try {
    const response = await axios.get('https://www.nseindia.com', { headers });
    const cookies = response.headers['set-cookie'];

    // Set cookies in the cookie jar
    cookies.forEach(cookie => {
      cookieJar.setCookieSync(cookie, 'https://www.nseindia.com');
    });
  } catch (error) {
    console.error('Error setting initial cookies:', error);
    return res.status(500).send('Server Error');
  }

  // Use the cookie jar for subsequent requests
  const url1 = `https://www.nseindia.com/api/option-chain-equities?symbol=${symbol}&date=${expiryDate}`;
  const url2 = `https://www.nseindia.com/api/option-chain-indices?symbol=${symbol}&date=${expiryDate}`;
  const url = (symbol === 'NIFTY' || symbol==='BANKNIFTY' || symbol ==='MIDCPNIFTY' || symbol==='FINNIFTY') ? url2 : url1;

  try {
    const response = await axios.get(url, {
      headers: {
        ...headers,
        cookie: cookieJar.getCookieStringSync('https://www.nseindia.com'),
      },
    });

    // Process the response and send the data back
    const { data } = response;
    const filteredData = data.filtered.data;
 
    const optionChain = {
      calls: [],
      puts: [],
      pcr: [],
    };

    filteredData.forEach((option) => {
      const callOption = {
        strikePrice: option.CE?.strikePrice,
        expiryDate: option.CE?.expiryDate,
        underlying: option.CE?.underlying,
        lastPrice: option.CE?.lastPrice,
        openInterest: option.CE?.openInterest,
        changeinOpenInterest: option.CE?.changeinOpenInterest,
        underlyingValue:option.CE?.underlyingValue
      };

      const putOption = {
        strikePrice: option.PE?.strikePrice,
        expiryDate: option.PE?.expiryDate,
        underlying: option.PE?.underlying,
        lastPrice: option.PE?.lastPrice,
        openInterest: option.PE?.openInterest,
        changeinOpenInterest: option.PE?.changeinOpenInterest,
        underlyingValue:option.PE?.underlyingValue
      };
      optionChain.calls.push(callOption);
      optionChain.puts.push(putOption);

      const ratio = (option.PE?.openInterest + 0.5) / (0.5 + option.CE?.openInterest);
      optionChain.pcr.push(ratio);
    });

   
    res.send(optionChain);
  } catch (error) {
    console.error('Error making request:', error);
    res.status(500).send('Server Error');
  }
  
  
  });

  router.post("/max-pain", async (req, res) => {
    const symbol = req.body.symbol;
    
    // Initialize a cookie jar
    const cookieJar = new tough.CookieJar();
  
    // Define headers with the cookie jar
    const headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept-Encoding': 'gzip, deflate, br',
      'Referer': 'https://www.nseindia.com/',
      'X-Requested-With': 'XMLHttpRequest',
      'Connection': 'keep-alive',
    };
  
    // Make the initial request to obtain cookies
    try {
      const response = await axios.get('https://www.nseindia.com', { headers });
      const cookies = response.headers['set-cookie'];
  
      // Set cookies in the cookie jar
      cookies.forEach(cookie => {
        cookieJar.setCookieSync(cookie, 'https://www.nseindia.com');
      });
    } catch (error) {
      console.error('Error setting initial cookies:', error);
      return res.status(500).send('Server Error');
    }
  
    // Use the cookie jar for subsequent requests
    const expiryDate = req.body.date;
    const url1 = `https://www.nseindia.com/api/option-chain-equities?symbol=${symbol}&date=${expiryDate}`;
    const url2 = `https://www.nseindia.com/api/option-chain-indices?symbol=${symbol}`;
    const url = (symbol === 'NIFTY' || symbol==='BANKNIFTY' || symbol ==='MIDCPNIFTY' || symbol==='FINNIFTY') ? url2 : url1;
  
    try {
      const response = await axios.get(url, {
        withCredentials: true,
        jar: cookieJar,
        headers: {
          ...headers,
          cookie: cookieJar.getCookieStringSync('https://www.nseindia.com'),
        },
      });
  
      // Process the response and send the data back
      const { data } = response;
      console.log(response)
      const filteredData = data.filtered.data;
      console.log(data)

      const callPain = [];
      const putPain = [];
      let maxpain = [];
  
      filteredData.forEach((option, index) => {
        const strikePrice = option.CE?.strikePrice;
  
        const slicedEntries = Object.entries(filteredData).slice(0, index);
  
        let pain = 0;
        slicedEntries.forEach((option) => {
          const callOi = option[1].CE?.openInterest;
          const strikeP = option[1].CE?.strikePrice;
  
          const difference = strikePrice - strikeP;
          pain = pain + callOi * difference;
        });
  
        callPain.push(pain);
      });
  
      const keys = Object.keys(filteredData);
      for (let i = keys.length - 1; i >= 0; i--) {
        const key = keys[i];
        const value = filteredData[key];
        const strikePrice = filteredData[key].PE?.strikePrice;
  
        const slicedEntries = Object.entries(filteredData).slice(i, keys.length);
        let pain = 0;
        slicedEntries.forEach((option) => {
          const putOi = option[1].PE?.openInterest;
          const strikeP = option[1].PE?.strikePrice;
  
          const difference = Math.abs(strikePrice - strikeP);
          pain = pain + putOi * difference;
        });
  
        putPain.push(pain);
      }
  
      const pPain = putPain.reverse();
  
      filteredData.forEach((option, index) => {
        const result = {
          strikePrice: option.CE.strikePrice,
          maxPain: (callPain[index] + pPain[index]) * 100,
        };
  
        maxpain.push(result);
      });
  
    
      res.send(maxpain);
    } catch (error) {
      console.error('Error making request:', error);
      res.status(500).send('Server Error');
    }
  });

  router.post("/kite", (req, res) => {
   
    const apiKey = "0bpuke0rhsjgq3lm";
    const redirectUri = "http://localhost:3000/scanner";

    const login_url = `https://kite.zerodha.com/connect/login?api_key=${apiKey}&v=3`;

    open(login_url);

    
  });


  // router.post("/upstox", async (req, res) => {
    // const totpCode = otplibAuthenticator.generate('7IUFMEYXZGOW2RDWQNOMMET3GV6URFYW' );

    // const options = new chrome.Options();
   

    // const driver = await new Builder()
    //   .forBrowser('chrome')
    //   .setChromeOptions(options) // Set to true for headless mode
    //   .build();

    // await driver.get(authUrl);



    // await driver.findElement(By.id('mobileNum')).sendKeys('7077376003');
    
    // await driver.findElement(By.id('getOtp')).click();

    // await driver.wait(until.elementLocated(By.id('otpNum')), 5000);

    // // Now, interact with the element

    // await driver.findElement(By.id('otpNum')).sendKeys(totpCode);

    // await driver.findElement(By.id('continueBtn')).click();

    // await driver.wait(until.elementLocated(By.id('pinCode')), 5000);

    // await driver.findElement(By.id('pinCode')).sendKeys('789123');

    // await driver.findElement(By.id('pinContinueBtn')).click();

    // res.status(200);

    // try {
      
    //   const browser = await puppeteer.launch({ headless: false });
    //   const page = await browser.newPage();
    // await page.setViewport({ width: 1550, height: 650});
    // // Navigate to the Upstox login page  
    // await page.goto(authUrl);

    // // Find and fill in the login form
    // await page.waitForSelector('#mobileNum');
    // await page.type('#mobileNum', '7077376003');
    // await page.click('#getOtp');
    
    // await page.waitForSelector('#otpNum');

    // const totpCode = otplibAuthenticator.generate('7IUFMEYXZGOW2RDWQNOMMET3GV6URFYW' );
    
    // console.log(totpCode)
    // await page.type('#otpNum', totpCode);

    // await page.click('#continueBtn');
    // // Wait for the PIN input field to appear
    // await page.waitForSelector('#pinCode');

    // // Enter the 6-digit PIN
    // await page.type('#pinCode', '789123'); // Replace '789123' with your actual 6-digit PIN

    // // Click the submit button
    // await page.click('#pinContinueBtn');

    // // Wait for the login to complete
    // // You may need to adjust the selector or add appropriate wait logic

    // // Extract the cookies
    // const cookies = await page.cookies();
    

    // // Close the browser
    // // await browser.close();
    // // Send the cookies back to your server
    // res.status(200).json({ cookies });
    
    // } catch (error) {
    //   console.error('Error during Upstox login:', error);
    //   res.status(500).json({ error: 'Internal Server Error' });
    // }
   
    // const apiKey = "0bpuke0rhsjgq3lm";
    // const redirectUri = "http://localhost:3000/scanner";

    // const login_url = `https://kite.zerodha.com/connect/login?api_key=${apiKey}&v=3`;
    
    // open(login_url);
    // res.send(authUrl)
    // open(authUrl)
  //   console.log(authUrl)
  //   res.json({ authUrl });
  // });

  router.post("/exit", (req, res) => {
    const type = req.body.exitTrade.type;
    const order = req.body.exitTrade.order;
    const triggerPrice = req.body.exitTrade.triggerPrice;
    const pro = req.body.exitTrade.pro;
    const quantity = req.body.exitTrade.quantity;
    const symbol = req.body.exitTrade.symbol;
    const exchange = req.body.exitTrade.exchange;
    const gtt = req.body.exitTrade.gttId  
    const stoploss = Number(req.body.exitTrade.stopLoss)
    const target =  Number(req.body.exitTrade.squareOff)
    const leftQuantity = req.body.exitTrade.left

    const orderParams = {
      exchange: exchange,
      tradingsymbol: symbol,
      quantity: quantity,
      transaction_type: type,
      order_type: order,
      product: pro,
    };

    console.log(orderParams);
  
    kite
      .placeOrder("regular", orderParams)
      .then(function (response) {
        console.log(response);
        const order_id = response.order_id;
        console.log(order_id);
  
        kite
          .getOrderHistory(order_id)
          .then(function (response) {
            console.log(response);
            const avgPrice = response[4].average_price;
            const quantity = response[4].quantity;
            const token = response[4].instrument_token;


            if (leftQuantity===0){

              kite.deleteGTT(gtt)
              .then(res=>console.log(res))
              .catch(err=>console.log(err))
      
            }
              
            else{
              function modifyGTTStopLoss(gttId) {
             
                const gttModificationParams = {
                  trigger_id:gttId,
                  trigger_type: 'two-leg',
                  tradingsymbol: symbol,
                  exchange: 'NSE',
                  trigger_values: [stoploss,target], 
                  last_price: avgPrice, 
                  orders: [
                    {
                      tradingsymbol: symbol,
                      exchange: 'NSE',
                      transaction_type: 'SELL',
                      quantity: leftQuantity,
                      product: pro,
                      order_type: 'MARKET',
                      price: stoploss,
                    },
                    {
                      tradingsymbol: symbol,
                      exchange: 'NSE',
                      transaction_type: 'SELL',
                      quantity: leftQuantity,
                      product: pro,
                      order_type: 'MARKET',
                      price: target,
                    }
                  
                  ],
                };
                
                console.log(gttModificationParams,'modify')
                kite.modifyGTT(gttId, gttModificationParams)
                .then((Response) => {
                  
                  console.log("Response:",Response);
               
                })
                .catch((error) => {
                  console.error("Error placing Target GTT Order:", error);
                });; 
              }
          
          
              modifyGTTStopLoss(gtt)
      
            }
      




            res.send({ avgPrice });
          })
          .catch(function (err) {
            console.error(err);
            res.status(500).send({ error: "Error fetching order history" });
          });
      })
      .catch(function (err) {
        console.error(err);
        res.status(500).send({ error: "Error placing order" });
      });
  });



  let updatedPnl={}
  router.post("/punch", (req, res) => {

    const exchange = req.body.trade.exchange
    const type = req.body.trade.type;
    const order = req.body.trade.order;
    const triggerPrice = req.body.trade.triggerPrice;
    const pro = req.body.trade.pro;
    const quantity = req.body.trade.quantity;
    const symbol = req.body.trade.symbol;
    var stoploss = Number(req.body.trade.stopLoss)
    const squareoff = Number(req.body.trade.squareOff)
    const trailingStopPercentage = Number(req.body.trade.trailingSL)
    const protectProfit = Number(req.body.trade.protectProfit)
    const timer = Number(req.body.trade.timer)
  
    var currPrice = null;
    var avgPrice = 0;
    const stocks = {};
    const instrument=[]
    const instrumentData={}
    let finalPnl ={}
    const tradeId = uuidv4();
    let signalSent = false;
    let stopSendingData = false;
    var gttId = null;

    console.log('trailing',trailingStopPercentage)
  ; // Adjust this percentage as needed
    let highestPrice = 0;
    let currentTrailingStopPrice = null // Initial stop-loss price
     // Replace with your two-leg GTT trigger ID


    const roll =req.body.roll
    console.log(symbol, pro, order, type, quantity);
    console.log(stoploss,squareoff)


    const orderParams = {
      exchange: exchange,
      tradingsymbol: symbol,
      quantity: quantity,
      transaction_type: type,
      order_type: order,
      product: pro,
     
    };
    console.log(orderParams)

    // if (triggerPrice) {
    //   orderParams.trigger_price = triggerPrice;
    // }
    

    kite
      .placeOrder("regular", orderParams)
      .then(function (response) {
        console.log(response);
        order_id =response.order_id
        console.log(order_id)
      
       kite.getOrderHistory(order_id)
        .then(function (response) {
          console.log(response);
          avgPrice = response[4].average_price;
          const quantity = response[4].quantity;
          const token = response[4].instrument_token
         

          if (!stocks[symbol]) {
            stocks[symbol] = {
              symbol,
              token,
              avgPrice,
              quantity
            };
          }


          const gttParams = {
            trigger_type: 'two-leg',
            tradingsymbol: symbol,
            exchange: exchange,
            trigger_values: [stoploss,squareoff], 
            last_price: avgPrice, 
            orders: [
              {
                tradingsymbol: symbol,
                exchange: exchange,
                transaction_type: 'SELL',
                quantity: quantity,
                product: pro,
                order_type: 'MARKET',
                price: stoploss,
              },
              {
                tradingsymbol: symbol,
                exchange: exchange,
                transaction_type: 'SELL',
                quantity: quantity,
                product: pro,
                order_type: 'MARKET',
                price: squareoff,
              }
            ],
          };


          

          kite.placeGTT(gttParams)
          .then((targetGTTResponse) => {
            gttId=targetGTTResponse.trigger_id
            console.log("Target GTT Order Response:", targetGTTResponse);
            console.log(gttId)
            res.send({tradeId,avgPrice,gttId});
            // Add any additional logic or handling for the target GTT order here
          })
          .catch((error) => {
            console.error("Error placing Target GTT Order:", error);
          });

    
        
          
        })
        .catch(function (err) {
          console.error(err);
        });

         
      })
      .catch(function (err) {
        console.error(err);
      });
                 
    const ticker = new KiteTicker({
      api_key: "0bpuke0rhsjgq3lm",
      access_token: access_token,
    });

      getQuote(["NSE:" + symbol]);

      function getQuote(instruments) {
        kite
          .getQuote(instruments)
          .then(function (response) {
            console.log(response);
            const token = response[instruments[0]].instrument_token;
            instrument.push(token)
            console.log('okkk')
            console.log(instrument)
            
            
    
            ticker.connect();
            ticker.on("ticks", onTicks);
            ticker.on("connect", subscribe);
           
            
            // ticker.on('disconnect', onDisconnect);
          })
          .catch(function (err) {
            console.log(err);
          });
      }  

       
    function sock (){
      console.log('hello')
      io.on("connection", (socket) => {
       
  
        console.log("Client connected");
  
        setInterval(()=>{
          pnlCal(symbol,socket)}, 2000);
  
        socket.on("disconnect", () => {
          console.log("Client disconnected");
        });
      });
  
    }

  
  
    
   sock()
   ticker.on('order_update', onTrade);

    function onTicks(ticks) {
      
      currPrice = ticks[0].last_price;
  
      
      ticks.forEach((tick) => {
        const instrumentToken = tick.instrument_token;
        const lastPrice = tick.last_price;
        const open = tick.ohlc.open

        const isTargetHit = lastPrice >= squareoff;
        const isStopLossHit = lastPrice <= stoploss;



        if ((isTargetHit || isStopLossHit) && !signalSent) {

          const hitType = isTargetHit ? 'target' : 'stopLoss';

          io.emit('tradeCompleted', { status: 'completed', tradeId: tradeId, tradeType: type, tradeSymbol: symbol ,hitType: hitType , roll:roll});
          signalSent = true
          stopSendingData = true;
          
          // ... (rest of your code)
        }

        
        if (lastPrice > highestPrice) {
        
          highestPrice = lastPrice;
          console.log(highestPrice)
        }


        const newTrailingStopPrice = highestPrice * (1 - trailingStopPercentage / 100);

        if (currentTrailingStopPrice === null) {
          // Initialize currentTrailingStopPrice on the first tick
          currentTrailingStopPrice = newTrailingStopPrice;
          stoploss=newTrailingStopPrice
        } else if (newTrailingStopPrice > currentTrailingStopPrice && !signalSent) {
          // Update the current trailing stop price only if it's initialized
          currentTrailingStopPrice = newTrailingStopPrice;
          stoploss=newTrailingStopPrice
          console.log(currentTrailingStopPrice)
          console.log('stoploss',stoploss);
          // Call a function to modify the stop-loss leg of the two-leg GTT order
          modifyGTTStopLoss(gttId, currentTrailingStopPrice);
        }
    
        const targetProfit = squareoff - avgPrice;

        // Define your protect profit percentage
        ; // Replace with your desired percentage
        
        // Calculate the protect profit threshold
        const protectProfitThreshold = avgPrice + (targetProfit * (protectProfit / 100));
        const newProtectProfit = avgPrice + (targetProfit *((protectProfit-10)/100))



        // Inside your onTicks function
        if (lastPrice >= protectProfitThreshold) {
          // Update the stop-loss to the current price
          modifyGTTStopLoss(gttId, newProtectProfit);
        }

           

        // Update instrument data with the latest last price
        instrumentData[instrumentToken] = {
          open:open,
          lastPrice: lastPrice,
        };

  
      })

    

    }

    function modifyGTTStopLoss(gttId, newStopLossPrice) {
      // Define the modification parameters for the stop-loss leg
      const gttModificationParams = {
        trigger_id:gttId,
        trigger_type: 'two-leg',
        tradingsymbol: symbol,
        exchange: 'NSE',
        trigger_values: [newStopLossPrice,squareoff], 
        last_price: avgPrice, 
        orders: [
          {
            tradingsymbol: symbol,
            exchange: 'NSE',
            transaction_type: 'SELL',
            quantity: quantity,
            product: pro,
            order_type: 'MARKET',
            price: newStopLossPrice,
          },
          {
            tradingsymbol: symbol,
            exchange: 'NSE',
            transaction_type: 'SELL',
            quantity: quantity,
            product: pro,
            order_type: 'MARKET',
            price: squareoff,
          }
        
        ],
      };
      // Call the modifyGTT method to modify the stop-loss leg
      kite.modifyGTT(gttId, gttModificationParams)
      .then((Response) => {
        
        console.log("Response:",Response);
        // Add any additional logic or handling for the target GTT order here
      })
      .catch((error) => {
        console.error("Error placing Target GTT Order:", error);
      });; // You'll need to implement the modifyGTT function
    }


    function subscribe() {
      
      ticker.subscribe(instrument);
      ticker.setMode(ticker.modeFull, instrument);
  
    }

    function onTrade(order) {
      console.log("holaaa amigos");
      console.log(order);
      if (order.status === 'COMPLETE' && order.transaction_type === 'SELL') { 
        io.emit('tradeCompleted', { 
          status: 'completed', 
          tradeId: tradeId, 
          tradeType: type, 
          tradeSymbol: symbol,
          hitType: 'stopLoss', // You may set the 'hitType' as needed
          roll: roll
        })
        signalSent = true;
      }
    
           
    }

    function onDisconnect(error) {
      console.log("Closed connection on disconnect", error);
    }



    function getPositions() {
      kite
        .getPositions()
        .then(function (response) {
          console.log(response);
          // const pnlData = response.day.map((item) => ({
          //   symbol: item.tradingsymbol,
          //   pnl: item.pnl,
          // }));

          response.day.forEach((item) => {
            const symbol = item.tradingsymbol;
            const avgPrice = item.average_price;
            const quantity = item.quantity;
            const token =item.instrument_token
            
            instrument.push(token)

            if (!stocks[symbol]) {
              stocks[symbol] = {
                symbol,
                token,
                avgPrice,
                quantity
              };
            } else {
              stocks[symbol].avgPrice += avgPrice;
              stocks[symbol].quantity += quantity;
              stocks[symbol].token += token;
            }
          });

          console.log(stocks)

          console.log(instrument)


         
        })
        .catch(function (err) {
          console.log(err);
        });
    }

    function pnlCal(symbol){
     
      if (Object.keys(stocks).length >0 && stocks[symbol]) {
      const token = stocks[symbol].token
      const quantity = stocks[symbol].quantity
     

      if( instrumentData[token]){
      
      
      const lastPrice = instrumentData[token].lastPrice
      const open = instrumentData[token].open

      const pnl = (lastPrice - avgPrice) * quantity;
          
      const day= ((lastPrice-open)/open)*100

      const update = {
        symbol:symbol,
        pnl:pnl,
        dayChange:day,
        avgPrice:avgPrice,
        quantity:quantity,
        ltp:lastPrice
       }
       
      finalPnl=update
      
      }
    
    }

    if (!stopSendingData) {
    io.emit('holdings',{tradeId,finalPnl})
  }
    }

  });

  router.post("/position", (req, res) => {
    var instrumentToken = null;
    var price = null;
    var avgPrice = null;
    var finalPnl = {};
    var pnlData =[]
    var instrumentSymbols=[]
    const instrumentData = {};


    kite
        .getHoldings()
        .then(function (response) {
         
           pnlData = response.map((item) => ({
            symbol: item.tradingsymbol,
            instrumentToken:item.instrument_token,
            pnl: item.pnl,
            quantity: item.quantity,
            ltp: item.last_price,
            dayChangePercentage: item.day_change_percentage,
            avgPrice: item.average_price,
          }));

          console.log(pnlData)
          pnlData.map((item)=>{
            instrumentSymbols.push(item.instrumentToken)
            }) 

          
          console.log(instrumentSymbols)
        })
        .catch(function (err) {
          console.log(err.response);
        });


    const ticker = new KiteTicker({
      api_key: "0bpuke0rhsjgq3lm",
      access_token: access_token,
    });

    ticker.connect();
    ticker.on("ticks", onTicks);
    ticker.on("connect", subscribe);
    ticker.on("order_update", onTrade);

    function onTicks(ticks) {
    
      
      ticks.forEach((tick) => {
        const instrumentToken = tick.instrument_token;
        const lastPrice = tick.last_price;
        const open = tick.ohlc.open
        // Update instrument data with the latest last price
        instrumentData[instrumentToken] = {
          open:open,
          lastPrice: lastPrice,
        };

      });

    

    }

    function subscribe() {
      var items = [instrumentToken];
      ticker.subscribe(instrumentSymbols);
      ticker.setMode(ticker.modeFull, instrumentSymbols);
    }

    function onTrade(order) {
      console.log("holaaadasd");

      if (order.status === "COMPLETE" && order.pnl !== 0) {
        var pnl = order.pnl;
        console.log("P&L:", pnl);
      }
    }

    // getHoldings();

    function getHoldings() {
      kite
        .getHoldings()
        .then(function (response) {
          const pnlData = response.map((item) => ({
            symbol: item.tradingsymbol,
            pnl: item.pnl,
            quantity: item.quantity,
            ltp: item.last_price,
            dayChangePercentage: item.day_change_percentage,
            avgPrice: item.average_price,
          }));
          res.send(pnlData);
        })
        .catch(function (err) {
          console.log(err.response);
        });
    }

    io.on("connection", (socket) => {
     
      console.log("Client connected");

      setInterval(()=>{
        pnlCalculation(socket)}, 2000);

      socket.on("disconnect", () => {
        console.log("Client disconnected");
      });
    });

    function getPositions() {
      kite
        .getPositions()
        .then(function (response) {
          const pnlData = response.net.map((item) => ({
            symbol: item.tradingsymbol,
            pnl: item.pnl,
          }));

          res.send(pnlData);
        })
        .catch(function (err) {
          console.log(err);
        });
    }

    function pnlCalculation(socket) {
      

      pnlData.map((item)=>{
        
        
          const price = item.avgPrice
          const symbol= item.symbol
          const instrument = item.instrumentToken
          const quantity = item.quantity
          const  dayChange = item.dayChangePercentage
          const avgPrice =item.avgPrice

          console.log(price,symbol,instrument,quantity)

         if (instrumentData[instrument]) {
          const lastPrice = instrumentData[instrument].lastPrice;
          const open = instrumentData[instrument].open
          const pnl = (lastPrice - price) * quantity;
          
          const day= ((lastPrice-open)/open)*100

          console.log(pnl)
          const update = {
            symbol:symbol,
            pnl:pnl,
            dayChange:day,
            avgPrice:avgPrice,
            quantity:quantity,
            ltp:lastPrice
           }
         
        finalPnl[symbol]=update

        }

      })
      
      socket.emit("update", finalPnl);
    }
  });



  router.post('/backtest', (req, res) => {
    // const pythonProcess = spawn('python', ['./config/strategy.py', JSON.stringify(req.body)]);
    
    const pythonProcess = spawn('python3', [path.join(__dirname, './config/strategy.py'), JSON.stringify(req.body)]);

    let dataString = '';
    pythonProcess.stdout.on('data', (data) => {
        dataString += data.toString();
        // console.log(dataString)
    });

    pythonProcess.on('close', (code) => {
        console.log(`Process exited with code ${code}`);
        res.json(dataString); // Send the Python script's response back to React
    });
});

// router.post('/backtest', (req, res) => {
//   const input = JSON.stringify(req.body);
//   const pythonProcess = spawn('python', ['./config/strategy.py', input]);

//   let dataString = '';
  
//   pythonProcess.stdout.on('data', (data) => {
//       dataString += data.toString();
//   });

//   pythonProcess.stderr.on('data', (data) => {
//       console.error(`stderr: ${data}`);
//   });

//   pythonProcess.on('close', (code) => {
//       console.log(`Process exited with code ${code}`);
//       try {
//           const responseJson = JSON.parse(dataString);
//           res.json(responseJson);
//       } catch (error) {
//           console.error('Error parsing JSON:', error);
//           res.status(500).json({ error: 'Failed to parse response from Python script.' });
//       }
//   });
// });


router.post('/optimize', (req, res) => {
  const { constraints, variableInputs ,goal, strategy} = req.body;

  console.log('Received constraints:', constraints);
  console.log('Received variableInputs:', variableInputs);
  console.log(goal)
 

  const pythonScriptPath = path.resolve(__dirname, 'config/optimise.py');
  const pythonProcess = spawn('python', [pythonScriptPath]);

  let dataString = '';

  pythonProcess.stdin.write(JSON.stringify({ constraints, variableInputs , goal ,strategy }));
  pythonProcess.stdin.end();

  pythonProcess.stdout.on('data', (data) => {
      dataString += data.toString();
      console.log(`stdout: ${data.toString()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
      console.error(`stderr: ${data.toString()}`);
  });

  pythonProcess.on('error', (error) => {
      console.error(`Error: ${error.message}`);
      res.status(500).send({ error: 'Internal Server Error' });
  });

  pythonProcess.on('close', (code) => {
      console.log(`Process exited with code ${code}`);
      if (code === 0) {
          res.json(JSON.parse(dataString)); // Send the Python script's response back to React
      } else {
          res.status(500).send({ error: 'Python script failed' });
      }
  });
});


  return router;
};
