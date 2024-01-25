-- This line is not strictly necessary as SQLite will create the file when you create a table
ATTACH DATABASE 'MDC.db' AS MDC;

-- create Album table
CREATE TABLE IF NOT EXISTS MDC.Album(
  album_id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  artist TEXT NOT NULL,
  Year INTEGER NOT NULL,
  FOREIGN KEY (Artist) REFERENCES Artist(ArtistID)
);


-- create table Artist
CREATE TABLE IF NOT EXISTS MDC.Artist(
  artist_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artist_name TEXT NOT NULL
);

-- create Customer table
CREATE TABLE IF NOT EXISTS MDC.Customer(
  customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
  customer_name TEXT NOT NULL,
  customer_address TEXT NOT NULL,
  customer_phone TEXT NOT NULL,
  city TEXT NOT NULL,
  county TEXT NOT NULL,
  post_code TEXT NOT NULL,
  country TEXT NOT NULL
);

-- create Employee table
CREATE TABLE IF NOT EXISTS MDC.Employee(
  employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
  employee_name TEXT NOT NULL,
  employee_address TEXT NOT NULL,
  employee_phone TEXT NOT NULL,
  employee_role TEXT NOT NULL,
  reports_to INTEGER NOT NULL,
  city TEXT NOT NULL,
  county TEXT NOT NULL,
  post_code TEXT NOT NULL,
  country TEXT NOT NULL
);

-- create Invoice table
CREATE TABLE IF NOT EXISTS MDC.Invoice(
  invoice_id INTEGER PRIMARY KEY AUTOINCREMENT,
  customer_id INTEGER NOT NULL,
  invoice_date TEXT NOT NULL,
  invoice_total INTEGER NOT NULL,
  FOREIGN KEY (customer_id) REFERENCES Customer(customer_id)
);

-- create InvoiceLine table
CREATE TABLE IF NOT EXISTS MDC.InvoiceLine(
  invoice_line_id INTEGER PRIMARY KEY AUTOINCREMENT,
  invoice_id INTEGER NOT NULL,
  track_id INTEGER NOT NULL,
  unit_price INTEGER NOT NULL,
  quantity INTEGER NOT NULL,
  FOREIGN KEY (invoice_id) REFERENCES Invoice(invoice_id),
  FOREIGN KEY (track_id) REFERENCES Track(track_id)
);

-- create MediaType table
CREATE TABLE IF NOT EXISTS MDC.MediaType(
  media_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
  media_type_name TEXT NOT NULL
);

-- create Playlist table
CREATE TABLE IF NOT EXISTS MDC.Playlist(
  playlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
  playlist_name TEXT NOT NULL
);

-- create PlaylistTrack table
CREATE TABLE IF NOT EXISTS MDC.PlaylistTrack(
  playlist_track_id INTEGER PRIMARY KEY AUTOINCREMENT,
  playlist_id INTEGER NOT NULL,
  track_id INTEGER NOT NULL,
  FOREIGN KEY (playlist_id) REFERENCES Playlist(playlist_id),
  FOREIGN KEY (track_id) REFERENCES Track(track_id)
);

-- create Track table
CREATE TABLE IF NOT EXISTS MDC.Track(
  track_id INTEGER PRIMARY KEY AUTOINCREMENT,
  track_name TEXT NOT NULL,
  album_id INTEGER NOT NULL,
  media_type_id INTEGER NOT NULL,
  genre_id INTEGER NOT NULL,
  composer TEXT NOT NULL,
  milliseconds INTEGER NOT NULL,
  bytes INTEGER NOT NULL,
  unit_price INTEGER NOT NULL,
  FOREIGN KEY (album_id) REFERENCES Album(album_id),
  FOREIGN KEY (media_type_id) REFERENCES MediaType(media_type_id),
  FOREIGN KEY (genre_id) REFERENCES Genre(genre_id)
);

-- create Genre table
CREATE TABLE IF NOT EXISTS MDC.Genre(
  genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
  genre_name TEXT NOT NULL
);

