require('dotenv').config();
const mysql = require('mysql');

const db = mysql.createConnection({
  host: 'bjbjotkpn4piwqplzpwn-mysql.services.clever-cloud.com',
  user: 'unr1tnyago7kvkrv',
  password: '4jkun8UayxYkgHocyj9Y',
  database: 'bjbjotkpn4piwqplzpwn',
  port: 3306
});


function handleDisconnect() {
  

  db.connect((err) => {
    if (err) {
      console.error('Error reconnecting to the MySQL database:', err);
      setTimeout(handleDisconnect, 2000); // Try to reconnect after 2 seconds
    } else {
      console.log('Reconnected to the MySQL database');
    }
  });

  db.on('error', (err) => {
    console.error('Database error:', err);
    if (err.code === 'PROTOCOL_CONNECTION_LOST') {
      handleDisconnect(); // Reconnect on connection loss
    } else {
      throw err;
    }
  });

  
}

handleDisconnect();
module.exports = db;