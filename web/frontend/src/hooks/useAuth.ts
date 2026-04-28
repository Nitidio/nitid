import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { getMe, login as apiLogin, register as apiRegister } from "../api/auth";
import type { User } from "../types";

export function useAuth() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const { data: user, isLoading } = useQuery<User | null>({
    queryKey: ["me"],
    queryFn: async () => {
      if (!localStorage.getItem("access_token")) return null;
      return getMe();
    },
    retry: false,
  });

  const isAuthenticated = !!user;

  async function login(username: string, password: string) {
    const token = await apiLogin(username, password);
    localStorage.setItem("access_token", token);
    await queryClient.invalidateQueries({ queryKey: ["me"] });
    navigate("/runs");
  }

  async function register(username: string, password: string) {
    await apiRegister(username, password);
    await login(username, password);
  }

  function logout() {
    localStorage.removeItem("access_token");
    queryClient.clear();
    navigate("/login");
  }

  return { user, isLoading, isAuthenticated, login, register, logout };
}
