require('dotenv').config();
const mysql = require('mysql');

// Use the Clever Cloud environment variables to configure the connection
const db = mysql.createConnection({
  host: 'bjbjotkpn4piwqplzpwn-mysql.services.clever-cloud.com',
  user:'unr1tnyago7kvkrv',
  password:'4jkun8UayxYkgHocyj9Y',
  database: 'bjbjotkpn4piwqplzpwn',
  port: 3306
});



// Connect to the database
db.connect((err) => {
  if (err) {
    console.error('Error connecting to the MySQL database:', err);
    return;
  }
  console.log('Connected to the MySQL database');
});

module.exports = db;
