require('dotenv').config();
const mysql = require('mysql');

let db;

function handleDisconnect() {
  db = mysql.createConnection({
    host: 'bjbjotkpn4piwqplzpwn-mysql.services.clever-cloud.com',
    user: 'unr1tnyago7kvkrv',
    password: '4jkun8UayxYkgHocyj9Y',
    database: 'bjbjotkpn4piwqplzpwn',
    port: 3306
  });

  db.connect((err) => {
    if (err) {
      console.error('Error connecting to MySQL database:', err);
      setTimeout(handleDisconnect, 2000); // Retry connection after 2 seconds
    } else {
      console.log('Connected to the MySQL database');
    }
  });

  db.on('error', (err) => {
    console.error('MySQL error:', err);
    if (err.code === 'PROTOCOL_CONNECTION_LOST') {
      console.log('MySQL connection lost, reconnecting...');
      handleDisconnect(); // Reconnect if connection lost
    } else {
      throw err;
    }
  });
}

// Initialize connection
handleDisconnect();

module.exports = {
  getConnection: () => db  // Export a function to get the current `db` instance
};
