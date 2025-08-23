const mysql = require('mysql2');

let db;

function handleDisconnect() {
  db = mysql.createConnection({
    host: '82.112.231.150', // VPS IP
    user: 'dbuser',
    password: 'Sid$igma3',
    database: 'kandles_db',
    port: 3306
  });

  db.connect(err => {
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
      handleDisconnect();
    } else {
      throw err;
    }
  });
}

handleDisconnect();

module.exports = db;
