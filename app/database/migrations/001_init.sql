-- NEXUS initial schema. Mirrors app/database/models.py; run in the Supabase SQL editor.

create table if not exists users (
    id          uuid primary key default gen_random_uuid(),
    email       text not null unique,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create table if not exists conversations (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid references users(id) on delete cascade,  -- NOT NULL once auth lands
    title       text not null,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);
create index if not exists ix_conversations_user_id on conversations(user_id);

create table if not exists messages (
    id               uuid primary key default gen_random_uuid(),
    conversation_id  uuid not null references conversations(id) on delete cascade,
    role             text not null constraint messages_role_check check (role in ('user', 'assistant')),
    content          text not null,
    created_at       timestamptz not null default now()
);
create index if not exists ix_messages_conversation_id on messages(conversation_id);

create table if not exists documents (
    id                  uuid primary key default gen_random_uuid(),
    user_id             uuid references users(id) on delete cascade,
    filename            text not null,
    file_type           text not null,
    source              text not null,
    status              text not null default 'pending',
    pinecone_namespace  text not null,
    metadata            jsonb not null default '{}',
    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now()
);
create index if not exists ix_documents_user_id on documents(user_id);

create table if not exists agent_runs (
    id               uuid primary key default gen_random_uuid(),
    conversation_id  uuid references conversations(id) on delete cascade,
    query            text not null,
    status           text not null,
    latency_ms       integer,
    tools_used       jsonb not null default '[]',
    created_at       timestamptz not null default now()
);
create index if not exists ix_agent_runs_conversation_id on agent_runs(conversation_id);

create table if not exists tool_calls (
    id            uuid primary key default gen_random_uuid(),
    agent_run_id  uuid not null references agent_runs(id) on delete cascade,
    tool_name     text not null,
    input         jsonb not null,
    output        jsonb,
    latency_ms    integer,
    status        text not null,
    created_at    timestamptz not null default now()
);
create index if not exists ix_tool_calls_agent_run_id on tool_calls(agent_run_id);

create table if not exists evaluations (
    id                 uuid primary key default gen_random_uuid(),
    question           text not null,
    answer             text not null,
    faithfulness       double precision,
    relevance          double precision,
    context_precision  double precision,
    context_recall     double precision,
    created_at         timestamptz not null default now()
);

-- Backend connects with the service role; block the public anon key from these tables.
alter table users          enable row level security;
alter table conversations  enable row level security;
alter table messages       enable row level security;
alter table documents      enable row level security;
alter table agent_runs     enable row level security;
alter table tool_calls     enable row level security;
alter table evaluations    enable row level security;
