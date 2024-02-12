CREATE EXTENSION IF NOT EXISTS vector;

-- Check if the extension was created
--SELECT * FROM pg_extension WHERE extname = 'vector';

-- if not, create manually