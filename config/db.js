const mysql = require('mysql');

let db;

function handleDisconnect() {
  // Create a brand new connection instance
  db = mysql.createConnection({
    host: 'bjbjotkpn4piwqplzpwn-mysql.services.clever-cloud.com',
    user: 'unr1tnyago7kvkrv',
    password: '4jkun8UayxYkgHocyj9Y',
    database: 'bjbjotkpn4piwqplzpwn',
    port: 3306
  });

  db.connect((err) => {
    if (err) {
      console.error('Error reconnecting to MySQL:', err);
      setTimeout(handleDisconnect, 2000);
    } else {
      console.log('Connected to MySQL');
    }
  });

  db.on('error', function (err) {
    console.error('Database error:', err);
    if (err.code === 'PROTOCOL_CONNECTION_LOST') {
      handleDisconnect(); // Reconnect on connection loss
    } else {
      throw err;
    }
  });
}

// start first connection
handleDisconnect();

module.exports = db;
