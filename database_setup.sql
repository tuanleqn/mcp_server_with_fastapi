-- Simple Chatbot Authentication Database Setup
-- This script creates tables following the specified Prisma schema
-- Run this in your existing PostgreSQL database

-- Drop existing tables if they exist
DROP TABLE IF EXISTS
    QUERY,
    CHAT,
    USERS
CASCADE;

-- USER, CHAT, QUERY tables remain as before
CREATE TABLE USERS (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    password VARCHAR(100),
    role VARCHAR(50)
);

CREATE TABLE CHAT (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    user_id INT REFERENCES USERS(id),
    tenant_id INT NOT NULL
);

CREATE TABLE QUERY (
    id INTEGER PRIMARY KEY,
    chat_id INT REFERENCES CHAT(id),
    query TEXT,
    answer TEXT,
    tenant_id INT NOT NULL
);

-- Insert sample users
INSERT INTO USERS (id, name, email, password, role) VALUES
(100000, 'Alice Johnson', 'alice@example.com', 'hashedpassword1', 'admin'),
(100001, 'Bob Smith', 'bob@example.com', 'hashedpassword2', 'user'),
(100002, 'Charlie Lee', 'charlie@example.com', 'hashedpassword3', 'user'),
(100003, 'Dana Wright', 'dana@example.com', 'hashedpassword4', 'moderator'),
(100004, 'Evan Kim', 'evan@example.com', 'hashedpassword5', 'user');

-- Insert sample chats
INSERT INTO CHAT (id, name, user_id, tenant_id) VALUES
(1, 'Project Alpha Planning', 100000, 1),
(2, 'Customer Technical Issues', 100001, 1),
(3, 'Feedback on New UI Design', 100002, 1),
(4, 'Moderation Strategy', 100003, 1),
(5, 'Billing & Subscription Queries', 100004, 1);

-- Insert sample queries
INSERT INTO QUERY (id, chat_id, query, answer, tenant_id) VALUES
(1, 1,
 'Can we outline the core milestones for Project Alpha? I think we should include design, development, and testing phases with rough timelines.',
 'Yes, let''s break it down into Design (2 weeks), Development (4 weeks), Testing (2 weeks), and Launch (1 week). I will draft a Gantt chart.',
 1),
(2, 1,
 'What are the risks associated with our current tech stack for Project Alpha?',
 'The main risks are scalability and vendor lock-in. We may want to look at containerization or moving to a microservices approach.',
 1),
(3, 2,
 'A customer is facing repeated timeouts while trying to submit forms on our web portal. Could this be a frontend or backend issue?',
 'Timeouts during form submission are most often backend-related. Check the API response time on the logs, especially around the /submit endpoint.',
 1),
(4, 2,
 'Is there any quick workaround for customers who are blocked from logging in after 3 failed attempts?',
 'Currently, they must wait 15 minutes. However, support can unlock accounts manually from the admin dashboard under "User Access Control".',
 1),
(5, 3,
 'Some users say the redesigned dashboard feels cluttered. Should we simplify it or offer a classic view option?',
 'Offering a toggle between "Classic" and "Modern" views could improve user satisfaction while preserving flexibility.',
 1),
(6, 4,
 'What guidelines should we follow when moderating sensitive content in community posts?',
 'Refer to the Community Policy document. For sensitive content, escalate to the Content Review Team and flag the post for internal review.',
 1),
(7, 5,
 'How do we handle users who accidentally downgrade their subscription but still expect premium access?',
 'We can offer a grace period of 7 days for such cases. Make sure to log this in the CRM and inform billing to verify eligibility.',
 1),
(8, 5,
 'A customer claims they were double-charged this month. Can we verify that from our end?',
 'Yes, check the Stripe logs under that user''s account. If confirmed, initiate a refund and mark the case with high priority.',
 1);
