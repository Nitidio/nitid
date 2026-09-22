import { Link } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import logo from "../assets/nitid-logo.png";

export function NavBar() {
  const { user, logout } = useAuth();

  return (
    <nav className="navbar">
      <Link to="/runs" className="navbar-brand">
        <img src={logo} alt="nitid" />
      </Link>
      <div className="navbar-links">
        <Link to="/runs">Runs</Link>
        <Link to="/runs/new">New Run</Link>
        <Link to="/datasets/convert">Convert Dataset</Link>
      </div>
      <div className="navbar-user">
        <span>{user?.username}</span>
        <button onClick={logout}>Logout</button>
      </div>
    </nav>
  );
}
