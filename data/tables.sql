-- Use a specific database file (e.g., "MDC.db")
-- This line is not strictly necessary as SQLite will create the file when you create a table
ATTACH DATABASE 'MDC.db' AS MDC;

-- Create a table for your data within the database
CREATE TABLE IF NOT EXISTS MDC.Employees (
    id INTEGER PRIMARY KEY,
    column1 TEXT,
    column2 TEXT
);

-- Insert sample data

INSERT INTO MDC.Employees (id, column1, column2) VALUES
    (1, 'John Doe', 'Manager'),
    (2, 'Jane Smith', 'Developer'),
    (3, 'Bob Johnson', 'Designer'),
    (4, 'Alice Brown', 'Analyst'),
    (5, 'Charlie Davis', 'Engineer'),
    (6, 'Eva White', 'Marketing Specialist'),
    (7, 'Frank Black', 'HR Manager'),
    (8, 'Grace Wilson', 'Sales Representative');
