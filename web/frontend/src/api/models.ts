import client from "./client";

export async function listModels(): Promise<string[]> {
  const { data } = await client.get<string[]>("/models");
  return data;
}

export async function getModelClasses(modelName: string): Promise<Record<string, string>> {
  const { data } = await client.get<Record<string, string>>(`/models/${modelName}/classes`);
  return data;
}
