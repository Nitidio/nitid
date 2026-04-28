import client from "./client";
import type { User } from "../types";

export async function register(username: string, password: string): Promise<User> {
  const { data } = await client.post<User>("/auth/register", { username, password });
  return data;
}

export async function login(username: string, password: string): Promise<string> {
  const params = new URLSearchParams({ username, password });
  const { data } = await client.post<{ access_token: string }>("/auth/login", params, {
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });
  return data.access_token;
}

export async function getMe(): Promise<User> {
  const { data } = await client.get<User>("/auth/me");
  return data;
}
